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
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from statsmodels import api as sm

from nomad.metainfo import (
    Quantity,
    Package,
    SubSection,
    Section,
)
from nomad.metainfo.metainfo import Category
from nomad.datamodel.data import (
    EntryData,
    ArchiveSection,
    EntryDataCategory,
)
from nomad.datamodel.metainfo.basesections import (
    MeasurementResult,
    CompositeSystemReference,
)
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
)
from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure

from nomad_measurements import (
    InSituMeasurement,
    ProcessReference,
)

m_package = Package(name="LayTec EpiTT Schema")


class IKZLayTecEpiTTCategory(EntryDataCategory):
    m_def = Category(label="IKZ LayTec EpiTT", categories=[EntryDataCategory])


class ReflectanceWavelengthTransient(PlotSection, ArchiveSection):
    m_def = Section(
        a_eln=dict(lane_width="600px"),
        label_quantity="wavelength",
    )
    wavelength = Quantity(
        type=np.dtype(np.float64),
        unit="nanometer",
        description="Reflectance Wavelength",
    )
    wavelength_name = Quantity(
        type=str,
        description="Name of Reflectance ",
    )
    raw_intensity = Quantity(
        type=np.dtype(np.float64),
        shape=["*"],
        description="Normalized reflectance wavelength",
    )
    total_line_number = Quantity(
        type=np.dtype(np.int64),
        description="""
        The total line number found in the LayTec EpiTT data file.
        This will help setting the correct autocorrelation starting point and autocorrelation period.
        """,
    )
    autocorrelation_starting_point = Quantity(
        type=np.int64,
        default=0,
        description="""
        Add this parameter and save to smoothen the reflectance trace.
        Starting point of the window chosen for the autocorrelaation function calculation,
        according to statsmodels.tsa.acf method from statsmodels package
        """,
        a_eln={"component": "NumberEditQuantity"},
    )
    autocorrelation_period = Quantity(
        type=np.int64,
        default=4500,
        description="""
        Add this parameter and save to smoothen the reflectance trace.
        Period of the window chosen for the autocorrelaation function calculation,
        according to statsmodels.tsa.acf method from statsmodels package
        """,
        a_eln={"component": "NumberEditQuantity"},
    )
    autocorrelated_intensity = Quantity(
        type=np.dtype(np.float64),
        description="""
        Normalized reflectance wavelength smoothed
        with a autocorrelation algorythm implemented in the statsmodels package.
        """,
        shape=["*"],
    )


class LayTecEpiTTMeasurementResult(MeasurementResult):
    """
    Add description
    """

    process_time = Quantity(
        type=np.dtype(np.float64),
        unit="seconds",
        shape=["*"],
    )
    pyrometer_temperature = Quantity(
        type=np.dtype(np.float64),
        description="PyroTemp transient. LayTec's TrueTemperature of the substrate surface --> Emissivity-corrected pyrometer ",
        unit="celsius",
        shape=["*"],
    )
    reflectance_wavelengths = SubSection(
        section_def=ReflectanceWavelengthTransient, repeats=True
    )


class MeasurementSettings(ArchiveSection):
    """
    Add description
    """

    module_name = Quantity(
        type=str,  #'Ring TT1 1',
        description="MODULE_NAME",
    )
    wafer_label = Quantity(
        type=str,  #'Al Zone 1 (Center)',
        description="WAFER_LABEL",
    )
    wafer_zone = Quantity(
        type=str,  #'Center',
        description="WAFER_ZONE",
    )
    wafer = Quantity(
        type=str,  #'1',
        description="WAFER",
    )
    runtype_ID = Quantity(
        type=str,  #'20',
        description="RUNTYPE_ID",
    )
    runtype_name = Quantity(
        type=str,  #'AlGa510mm90',
        description="RUNTYPE_NAME",
    )


class LayTecEpiTTMeasurement(InSituMeasurement, PlotSection, EntryData):
    """
    LayTec's EpiTT is an emissivity-corrected pyrometer and
    reflectance measurement for in-situ measurement during
    growth processes (https://www.laytec.de/epitt)
    """

    m_def = Section(
        a_eln={"lane_width": "600px", "hide": ["steps"]},
        categories=[IKZLayTecEpiTTCategory],
        label="EpiTT Measurement",
        a_template=dict(
            instruments=[dict(name="LayTec_EpiTT", lab_id="LayTec_EpiTT_MOVPE_Ga2O3")],
        ),
    )
    description = Quantity(
        type=str,
        description="description",
        a_eln=ELNAnnotation(
            component="StringEditQuantity",
            label="Notes",
        ),
    )
    method = Quantity(
        type=str, description="Method used to collect the data", default="LayTec_EpiTT"
    )
    location = Quantity(
        type=str,
        description="""
        The location of the process in longitude, latitude.
        """,
        default="52.431685, 13.526855",  # IKZ coordinates
    )
    data_file = Quantity(
        type=str,
        description="Data file containing the EpiTT data (*.dat)",
        a_eln=dict(
            component="FileEditQuantity",
        ),
    )
    process = SubSection(
        section_def=ProcessReference,
        description="A reference to the process during which the measurement occurred.",
        label="growth_process",
    )
    measurement_settings = SubSection(section_def=MeasurementSettings)
    results = SubSection(
        section_def=LayTecEpiTTMeasurementResult,
        # repeats=True,
    )

    def normalize(self, archive, logger):
        super(LayTecEpiTTMeasurement, self).normalize(archive, logger)
        logger.info("Executed LayTecEpiTTMeasurement normalizer.")

        # reference the growth process entry
        if self.process.name:
            self.process.normalize(archive, logger)
            logger.info("Executed LayTecEpiTTMeasurement.process normalizer.")
            if hasattr(self.process.reference, "grown_sample"):
                sample_list = []
                sample_list.append(
                    CompositeSystemReference(
                        lab_id=self.process.reference.grown_sample.lab_id,
                    ),
                )
                self.samples = sample_list
                self.samples[0].normalize(archive, logger)
            else:
                logger.warning(
                    "No grown_sample found in GrowthMovpe2.grown_sample.\
                     No sample is referenced in LayTecEpiTTMeasurement. \
                     Please upload a growth process file and reprocess."
                )

        # noise smoothening with autocorrelated function
        for trace in self.results[0].reflectance_wavelengths:
            trace.total_line_number = len(trace.raw_intensity)
            if (
                trace.autocorrelation_period is not None
                and trace.autocorrelation_starting_point is not None
            ):
                trace.autocorrelated_intensity = sm.tsa.acf(
                    trace.raw_intensity[trace.autocorrelation_starting_point :],
                    nlags=trace.autocorrelation_period,
                    fft=False,
                )

        # plots
        if self.results[0]:
            overview_fig = make_subplots(
                rows=2, cols=1, subplot_titles=["Reflectance", "Temperature"]
            )
            temperature_figure = px.scatter(
                x=self.results[0].process_time,
                y=self.results[0].pyrometer_temperature,
                # color=self.results[0].pyrometer_temperature,
                title="Temp.",
            )
            temperature_figure.update_traces(mode="markers", marker={"size": 2})
            temperature_figure.update_xaxes(title_text="Time [s]")
            temperature_figure.update_yaxes(title_text="Temperature [°C]")
            overview_fig.add_trace(temperature_figure.data[0], row=2, col=1)
            multi_reflec_fig = make_subplots(rows=1, cols=1)
            for i, _ in enumerate(self.results[0].reflectance_wavelengths):
                reflec_fig = go.Figure(
                    # config={"displayModeBar": True, "scrollZoom": True}
                )
                reflec_fig.add_trace(
                    go.Scatter(
                        x=self.results[0].process_time.magnitude,
                        y=self.results[0].reflectance_wavelengths[i].raw_intensity,
                        mode="lines+markers",
                        line={"width": 2},
                        marker={"size": 2},
                        name=f"{self.results[0].reflectance_wavelengths[i].wavelength.magnitude} nm",
                    )
                )
                multi_reflec_fig.add_trace(
                    go.Scatter(
                        x=self.results[0].process_time.magnitude,
                        y=self.results[0].reflectance_wavelengths[i].raw_intensity,
                        mode="lines+markers",
                        line={"width": 2},
                        marker={"size": 2},
                        name=f"{self.results[0].reflectance_wavelengths[i].wavelength.magnitude} nm",
                    )
                )
                if (
                    self.results[0].reflectance_wavelengths[i].autocorrelated_intensity
                    is not None
                    and self.results[0]
                    .reflectance_wavelengths[i]
                    .autocorrelated_intensity.any()
                ):
                    reflec_fig.add_trace(
                        go.Scatter(
                            x=self.results[0].process_time.magnitude,
                            y=self.results[0]
                            .reflectance_wavelengths[i]
                            .autocorrelated_intensity,
                            mode="lines+markers",
                            line={"width": 2},
                            marker={"size": 2},
                            name=f"Autocorr. {self.results[0].reflectance_wavelengths[i].wavelength.magnitude} nm",
                        )
                    )
                    multi_reflec_fig.add_trace(
                        go.Scatter(
                            x=self.results[0].process_time.magnitude,
                            y=self.results[0]
                            .reflectance_wavelengths[i]
                            .autocorrelated_intensity,
                            mode="lines+markers",
                            line={"width": 2},
                            marker={"size": 2},
                            name=f"Autocorr. {self.results[0].reflectance_wavelengths[i].wavelength.magnitude} nm",
                        )
                    )
                reflec_fig.update_xaxes(title_text="Time [s]")
                reflec_fig.update_yaxes(title_text="Reflectance")
                reflec_fig.update_layout(
                    height=500,
                    # width=800,
                    showlegend=True,
                )
                self.results[0].reflectance_wavelengths[i].figures = [
                    PlotlyFigure(
                        label=f"{self.results[0].reflectance_wavelengths[i].wavelength.magnitude} nm",
                        index=1,
                        figure=reflec_fig.to_plotly_json(),
                    )
                ]
            for trace in multi_reflec_fig.data:
                overview_fig.add_trace(trace, row=1, col=1)

            overview_fig.update_layout(
                # title_text="Reflectance",
                # showlegend=True,
                # legend=dict(
                #     x=0.2,  # Adjust these values to place the legend where you want
                #     y=0.2,
                #     bgcolor="rgba(255, 255, 255, 0)",  # Transparent background
                #     bordercolor="rgba(255, 255, 255, 0)",  # Transparent border
                height=2500,
                # width=800,
                showlegend=True,
            )
            # overview_fig.add_annotation(
            #     x=self.results[0].process_time.magnitude[1000],
            #     y=self.results[0].reflectance_wavelengths[i].raw_intensity[1000],
            #     xref="x",
            #     yref="y",
            #     text="Annotation Text",
            #     showarrow=True,
            #     arrowhead=7,
            #     ax=0,
            #     ay=-40,
            # )

            self.figures = [
                PlotlyFigure(label="figure 1", figure=overview_fig.to_plotly_json())
            ]


#            if self.process.reference:
# with archive.m_context.raw_file(self.process.reference, 'r') as process_file:
#     process_dict = yaml.safe_load(process_file)
#     updated_dep_control['data']['grown_sample'] = GrownSamples(
#             reference=f"../uploads/{archive.m_context.upload_id}/archive/{hash(archive.metadata.upload_id, sample_filename)}#data",
#         ).m_to_dict()

m_package.__init_metainfo__()
