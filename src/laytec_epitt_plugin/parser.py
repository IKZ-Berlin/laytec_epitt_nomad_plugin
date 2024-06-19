#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
import math
import json
import yaml
import pandas as pd
from datetime import datetime

from nomad.units import ureg

from nomad.utils import hash
from nomad.datamodel import EntryArchive
from nomad.metainfo import Quantity, Section
from nomad.parsing import MatchingParser
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
)
from nomad.datamodel.data import (
    EntryData,
)
from nomad.datamodel.datamodel import EntryArchive, EntryMetadata

from nomad_measurements import ProcessReference
from laytec_epitt_plugin.schema import (
    LayTecEpiTTMeasurement,
    IKZLayTecEpiTTCategory,
    ReflectanceWavelengthTransient,
    LayTecEpiTTMeasurementResult,
    MeasurementSettings,
    RefractiveIndex,
)


def get_reference(upload_id, entry_id):
    return f"../uploads/{upload_id}/archive/{entry_id}"


def get_entry_id(upload_id, filename):
    from nomad.utils import hash

    return hash(upload_id, filename)


def get_hash_ref(upload_id, filename):
    return f"{get_reference(upload_id, get_entry_id(upload_id, filename))}#data"


def nan_equal(a, b):
    """
    Compare two values with NaN values.
    """
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    elif isinstance(a, dict) and isinstance(b, dict):
        return dict_nan_equal(a, b)
    elif isinstance(a, list) and isinstance(b, list):
        return list_nan_equal(a, b)
    else:
        return a == b


def list_nan_equal(list1, list2):
    """
    Compare two lists with NaN values.
    """
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if not nan_equal(a, b):
            return False
    return True


def dict_nan_equal(dict1, dict2):
    """
    Compare two dictionaries with NaN values.
    """
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1:
        if not nan_equal(dict1[key], dict2[key]):
            return False
    return True


def create_archive(
    entry_dict, context, filename, file_type, logger, *, overwrite: bool = False
):
    from nomad.datamodel.context import ClientContext

    if isinstance(context, ClientContext):
        return None
    if context.raw_path_exists(filename):
        with context.raw_file(filename, "r") as file:
            existing_dict = yaml.safe_load(file)
    if context.raw_path_exists(filename) and not dict_nan_equal(
        existing_dict, entry_dict
    ):
        logger.error(
            f"{filename} archive file already exists. "
            f"You are trying to overwrite it with a different content. "
            f"To do so, remove the existing archive and click reprocess again."
        )
    if (
        not context.raw_path_exists(filename)
        or existing_dict == entry_dict
        or overwrite
    ):
        with context.raw_file(filename, "w") as newfile:
            if file_type == "json":
                json.dump(entry_dict, newfile)
            elif file_type == "yaml":
                yaml.dump(entry_dict, newfile)
        context.upload.process_updated_raw_file(filename, allow_modify=True)

    return get_hash_ref(context.upload_id, filename)


class RawFileLayTecEpiTT(EntryData):
    """
    Contains the raw file from LayTecEpiTT in situ monitoring
    """

    m_def = Section(categories=[IKZLayTecEpiTTCategory])
    measurement = Quantity(
        type=LayTecEpiTTMeasurement,
        a_eln=ELNAnnotation(
            component="ReferenceEditQuantity",
        ),
    )


class EpiTTParser(MatchingParser):
    def parse(self, mainfile: str, archive: EntryArchive, logger) -> None:
        data_file = mainfile.split("/")[-1]
        data_file_with_path = mainfile.split("raw/")[-1]
        measurement_data = LayTecEpiTTMeasurement()
        measurement_data.measurement_settings = MeasurementSettings()
        # .m_from_dict(LayTecEpiTTMeasurement.m_def.a_template)
        measurement_data.data_file = data_file_with_path

        def parse_epitt_data(file):
            line = file.readline().strip()
            parameters = {}
            header = []
            while (
                line.startswith(
                    (
                        "##",
                        "!",
                    )
                )
                or line.strip() == ""
            ):
                match = re.match(r"##(\w+)\s*=\s*(.*)", line.strip())
                if match:
                    parameter_name = match.group(1)
                    parameter_value = match.group(2)
                    if parameter_name == "YUNITS":
                        yunits = parameter_value.split("\t")
                        parameters[parameter_name] = yunits
                    else:
                        parameters[parameter_name] = parameter_value
                line = file.readline().strip()
            header = line.split("\t")
            data_in_df = pd.read_csv(file, sep="\t", names=header, skipfooter=1)
            return parameters, data_in_df

        if measurement_data.data_file:
            with archive.m_context.raw_file(data_file_with_path) as file:
                epitt_data = parse_epitt_data(file)
                name_string = ""
                paramters_for_name = [
                    "RUN_ID",
                    "RUNTYPE_ID",
                    "RUNTYPE_NAME",
                    "MODULE_NAME",
                    "WAFER_LABEL",
                    "WAFER",
                ]
                for p in paramters_for_name:
                    if p in epitt_data[0].keys():
                        name_string += "_" + epitt_data[0][p]
                if name_string != "":
                    measurement_data.name = name_string[1:]
                    measurement_data.lab_id = name_string[1:]
                if "TIME" in epitt_data[0].keys():
                    measurement_data.datetime = datetime.strptime(
                        epitt_data[0]["TIME"], "%Y-%m-%d-%H-%M-%S"
                    )  #'2020-08-27-11-11-30',
                measurement_data.measurement_settings = MeasurementSettings()  # ?
                if "MODULE_NAME" in epitt_data[0].keys():
                    measurement_data.measurement_settings.module_name = epitt_data[0][
                        "MODULE_NAME"
                    ]
                if "WAFER_LABEL" in epitt_data[0].keys():
                    measurement_data.measurement_settings.wafer_label = epitt_data[0][
                        "WAFER_LABEL"
                    ]
                if "WAFER_ZONE" in epitt_data[0].keys():
                    measurement_data.measurement_settings.wafer_zone = epitt_data[0][
                        "WAFER_ZONE"
                    ]
                if "WAFER" in epitt_data[0].keys():
                    measurement_data.measurement_settings.wafer = epitt_data[0]["WAFER"]
                # if "RUN_ID" in epitt_data[0].keys():
                #    self.run_ID = epitt_data[0]["RUN_ID"]
                if "RUNTYPE_ID" in epitt_data[0].keys():
                    measurement_data.measurement_settings.runtype_ID = epitt_data[0][
                        "RUNTYPE_ID"
                    ]
                if "RUNTYPE_NAME" in epitt_data[0].keys():
                    measurement_data.measurement_settings.runtype_name = epitt_data[0][
                        "RUNTYPE_NAME"
                    ]
                # measurement_data.time_transient = epitt_data[1]["BEGIN"]
                process = ProcessReference()
                process.lab_id = epitt_data[0]["RUN_ID"]
                process.normalize(archive, logger)
                measurement_data.process = process
                results = LayTecEpiTTMeasurementResult()
                results.process_time = epitt_data[1]["BEGIN"]
                results.pyrometer_temperature = epitt_data[1]["PyroTemp"]
                results.reflectance_wavelengths = []
                for wl, datacolname in zip(
                    ["REFLEC_WAVELENGTH", "PYRO_WAVELENGTH", "WHITE_WAVELENGTH"],
                    ["DetReflec", "RLo", "DetWhite"],
                ):
                    if wl in epitt_data[0].keys():
                        spectrum = epitt_data[1][datacolname]
                        transient_object = ReflectanceWavelengthTransient()
                        transient_object.wavelength = (
                            float(epitt_data[0][wl])
                            * ureg("nanometer").to("meter").magnitude
                        )
                        transient_object.name = str(
                            int(
                                round(
                                    transient_object.wavelength.to(
                                        "nanometer"
                                    ).magnitude
                                )
                            )
                        )
                        transient_object.rawfile_column_header = wl
                        transient_object.raw_intensity = spectrum / spectrum[0]

                        # MANUAL refractive index assignment (until we have ELLIPSOMETRY data in the archive)
                        refractive_index = 1.0
                        if transient_object.name == "633":
                            # ELLIPSOMETRY measurement at 800 degree celsius
                            refractive_index = 1.95
                        elif transient_object.name == "405":
                            # ELLIPSOMETRY measurement at 800 degree celsius
                            refractive_index = 2.02
                        elif transient_object.name == "951":
                            # ELLIPSOMETRY measurement at 800 degree celsius
                            refractive_index = 1.93
                        if not getattr(transient_object, "refractive_index"):
                            transient_object.refractive_index = RefractiveIndex(
                                value=refractive_index
                            )

                        # smoothed_intesity is calculated in the LayTecEpiTTMeasurement normalizer
                        results.reflectance_wavelengths.append(transient_object)
                measurement_data.results = [results]
            filetype = "yaml"
            filename = f"{data_file[:-4]}_measurement.archive.{filetype}"
            measurement_archive = EntryArchive(
                data=measurement_data,
                m_context=archive.m_context,
                metadata=EntryMetadata(upload_id=archive.m_context.upload_id),
            )
            create_archive(
                measurement_archive.m_to_dict(),
                archive.m_context,
                filename,
                filetype,
                logger,
            )
        archive.data = RawFileLayTecEpiTT(
            measurement=f"../uploads/{archive.m_context.upload_id}/archive/{hash(archive.m_context.upload_id, filename)}#data"
        )
        archive.metadata.entry_name = data_file + " in situ measurement file"
