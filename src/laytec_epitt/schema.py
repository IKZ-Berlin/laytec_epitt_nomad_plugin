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
from scipy.signal import find_peaks

from nomad.metainfo import (
    Quantity,
    Package,
    MEnum,
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
    m_def = Category(label="LayTec EpiTT", categories=[EntryDataCategory])


class RefractiveIndex(ArchiveSection):
    reference = Quantity(
        type=ArchiveSection,
        description="A reference to a Ellipsometry measurement to obtain refractive index.",
        a_eln=ELNAnnotation(
            component="ReferenceEditQuantity",
        ),
    )
    value = Quantity(
        type=np.dtype(np.float64),
        a_eln=ELNAnnotation(
            component="NumberEditQuantity",
        ),
    )


class FindPeaksParameters(ArchiveSection):
    """
    Parameters for the find_peaks function from scipy.signal
    """

    minimum_peaks_distance = Quantity(
        type=np.dtype(np.float64),
        description="""
        Peak-to-peak distance of the reflectance oscillation,
        it is used for automatic peak recognition to calculate the growth rate.
        It will be injected as the distance parameter in the scipy.signal.find_peaks method.
        """,
        a_eln=ELNAnnotation(
            component="NumberEditQuantity",
        ),
    )
    first_peak_index = Quantity(
        type=int,
        description="""
        the index of the first maximum in the maxima vector to be considered for growth rate calculation.
        """,
        a_eln=ELNAnnotation(
            component="NumberEditQuantity",
        ),
    )
    last_peak_index = Quantity(
        type=int,
        description="""
        the index of the last maximum in the maxima vector to be considered for growth rate calculation.
        """,
        a_eln=ELNAnnotation(
            component="NumberEditQuantity",
        ),
    )
    peaks_abscissa = Quantity(
        type=np.dtype(np.float64),
        description="Positions of the peaks in the reflectance trace.",
        shape=["*"],
        unit="seconds",
        a_eln=dict(
            component="NumberEditQuantity",
            defaultDisplayUnit="second",
        ),
    )
    peaks_ordinate = Quantity(
        type=np.dtype(np.float64),
        description="Peak-to-peak distance of the reflectance oscillation.",
        shape=["*"],
    )
    first_valley_index = Quantity(
        type=int,
        description="""
        the index of the first minimum in the minima vector to be considered for growth rate calculation.
        """,
        a_eln=ELNAnnotation(
            component="NumberEditQuantity",
        ),
    )
    last_valley_index = Quantity(
        type=int,
        description="""
        the index of the last minimum in the minima vector to be considered for growth rate calculation.
        """,
        a_eln=ELNAnnotation(
            component="NumberEditQuantity",
        ),
    )
    valleys_abscissa = Quantity(
        type=np.dtype(np.float64),
        description="Positions of the valleys in the reflectance trace.",
        shape=["*"],
        unit="seconds",
        a_eln=dict(
            component="NumberEditQuantity",
            defaultDisplayUnit="second",
        ),
    )
    valleys_ordinate = Quantity(
        type=np.dtype(np.float64),
        description="Valley-to-valley distance of the reflectance oscillation.",
        shape=["*"],
    )


class GrowthRate(ArchiveSection):
    """
    Add description
    """

    reflectance_trace = Quantity(
        type=MEnum(
            "Raw",
            "Smoothed",
            "Autocorrelated",
        ),
        a_eln=ELNAnnotation(
            component="RadioEnumEditQuantity",
        ),
        description="Select the reflectance trace to calculate the growth rate.",
    )
    growth_period = Quantity(
        type=np.dtype(np.float64),
        unit="second",
        description="""
        averaged peak-to-peak (and valley-to-valley)
        distance of the reflectance oscillation at the selected wavelength
        in the reflectance vs. time plot.
        """,
        a_eln=ELNAnnotation(
            component="NumberEditQuantity",
            defaultDisplayUnit="second",
        ),
    )
    recalculate_on_save = Quantity(
        type=MEnum(
            "Yes",
            "No",
        ),
        description="When saving the section, recalculate the growth rate",
        a_eln=ELNAnnotation(
            component="RadioEnumEditQuantity",
        ),
    )
    growth_rate = Quantity(
        type=np.dtype(np.float64),
        unit="meter/second",
        a_eln=ELNAnnotation(
            component="NumberEditQuantity",
            defaultDisplayUnit="nm/second",
        ),
    )
    peaks_identification = SubSection(section_def=FindPeaksParameters)


class ReflectanceWavelengthTransient(PlotSection, ArchiveSection):
    m_def = Section(
        label_quantity="name",
    )
    name = Quantity(
        type=str,
        description="Name of the reflectance trace",
        a_eln=ELNAnnotation(
            component="StringEditQuantity",
        ),
    )
    wavelength = Quantity(
        type=np.dtype(np.float64),
        unit="meter",
        description="Reflectance Wavelength",
        a_eln=dict(
            component="NumberEditQuantity",
            defaultDisplayUnit="nanometer",
        ),
    )
    rawfile_column_header = Quantity(
        type=str,
        description="Name of Reflectance ",
    )
    autocorrelation_starting_point = Quantity(
        type=np.int64,
        description="""
        Add this parameter and save to smoothen the reflectance trace.
        Starting point of the window chosen for the autocorrelaation function calculation,
        according to statsmodels.tsa.acf method from statsmodels package
        """,
        a_eln={"component": "NumberEditQuantity"},
    )
    autocorrelation_period = Quantity(
        type=np.int64,
        description="""
        Add this parameter and save to smoothen the reflectance trace.
        Period of the window chosen for the autocorrelaation function calculation,
        according to statsmodels.tsa.acf method from statsmodels package
        """,
        a_eln={"component": "NumberEditQuantity"},
    )
    raw_intensity = Quantity(
        type=np.dtype(np.float64),
        shape=["*"],
        description="Normalized reflectance wavelength",
    )
    smoothed_intensity = Quantity(
        type=np.dtype(np.float64),
        shape=["*"],
        description="Normalized and smoothed reflectance wavelength. The smoothing is done with a moving average filter from Pandas package [.rolling(30).mean()].",
        a_plotly_express={
            "method": "line",
            "x": "#/data/results/0/process_time",
            "y": "#smoothed_intensity",
        },
    )
    autocorrelated_intensity = Quantity(
        type=np.dtype(np.float64),
        description="""
        Normalized reflectance wavelength processed
        with a autocorrelation algorythm implemented in the statsmodels package.
        """,
        shape=["*"],
        unit="seconds",
    )
    refractive_index = SubSection(section_def=RefractiveIndex)
    growth_rate = SubSection(section_def=GrowthRate)


class LayTecEpiTTMeasurementResult(MeasurementResult):
    """
    Add description
    """

    m_def = Section(
        a_eln=ELNAnnotation(
            lane_width="300px",
        ),
    )

    process_time = Quantity(
        type=np.dtype(np.float64),
        unit="second",
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
                    "No grown_sample found in GrowthMovpe2.grown_sample. "
                    + "No sample is referenced in LayTecEpiTTMeasurement. "
                    + "Please upload a growth process file and reprocess."
                )

        if self.results[0]:
            # noise smoothening with moving average
            for trace in self.results[0].reflectance_wavelengths:
                if not getattr(trace, "autocorrelation_starting_point"):
                    setattr(trace, "autocorrelation_starting_point", 0)
                if not getattr(trace, "autocorrelation_period"):
                    setattr(trace, "autocorrelation_period", len(trace.raw_intensity))
                start = trace.autocorrelation_starting_point
                period = trace.autocorrelation_period
                if period is not None and start is not None:
                    trace.smoothed_intensity = (
                        pd.Series(trace.raw_intensity[start : (start + period)])
                        .rolling(30)
                        .mean()
                    )

            # noise smoothening with autocorrelated function
            for trace in self.results[0].reflectance_wavelengths:
                if not getattr(trace, "autocorrelation_starting_point"):
                    setattr(trace, "autocorrelation_starting_point", 0)
                if not getattr(trace, "autocorrelation_period"):
                    setattr(trace, "autocorrelation_period", len(trace.raw_intensity))
                start = trace.autocorrelation_starting_point
                period = trace.autocorrelation_period
                if period is not None and start is not None:
                    trace.autocorrelated_intensity = sm.tsa.acf(
                        trace.raw_intensity[start : (start + period)],
                        nlags=period,
                        fft=False,
                    )

                # growth rate calculation
                refractive_index = getattr(
                    getattr(trace, "refractive_index", None), "value", None
                )
                if not getattr(trace, "growth_rate"):
                    setattr(trace, "growth_rate", GrowthRate())
                if not getattr(trace.growth_rate, "peaks_identification"):
                    setattr(
                        trace.growth_rate, "peaks_identification", FindPeaksParameters()
                    )
                if not getattr(
                    trace.growth_rate.peaks_identification,
                    "minimum_peaks_distance",
                    None,
                ):
                    if trace.name in ["951", "950"]:
                        setattr(
                            trace.growth_rate.peaks_identification,
                            "minimum_peaks_distance",
                            500,
                        )
                    if trace.name in ["633"]:
                        setattr(
                            trace.growth_rate.peaks_identification,
                            "minimum_peaks_distance",
                            300,
                        )
                    if trace.name in ["405"]:
                        setattr(
                            trace.growth_rate.peaks_identification,
                            "minimum_peaks_distance",
                            200,
                        )
                # fill some default values
                if not getattr(
                    trace.growth_rate.peaks_identification, "first_peak_index"
                ):
                    setattr(
                        trace.growth_rate.peaks_identification, "first_peak_index", 0
                    )
                if not getattr(
                    trace.growth_rate.peaks_identification, "last_peak_index"
                ):
                    setattr(
                        trace.growth_rate.peaks_identification, "last_peak_index", 4
                    )
                if not getattr(
                    trace.growth_rate.peaks_identification, "first_valley_index"
                ):
                    setattr(
                        trace.growth_rate.peaks_identification, "first_valley_index", 1
                    )
                if not getattr(
                    trace.growth_rate.peaks_identification, "last_valley_index"
                ):
                    setattr(
                        trace.growth_rate.peaks_identification, "last_valley_index", 5
                    )
                if not getattr(trace.growth_rate, "reflectance_trace"):
                    setattr(trace.growth_rate, "reflectance_trace", "Smoothed")
                if getattr(trace.growth_rate, "reflectance_trace") == "Raw":
                    chosen_trace = trace.raw_intensity
                if getattr(trace.growth_rate, "reflectance_trace") == "Smoothed":
                    chosen_trace = trace.smoothed_intensity
                if getattr(trace.growth_rate, "reflectance_trace") == "Autocorrelated":
                    chosen_trace = trace.autocorrelated_intensity
                # find peaks
                peaks_indices, _ = find_peaks(
                    chosen_trace,
                    distance=trace.growth_rate.peaks_identification.minimum_peaks_distance,
                )
                # find valleys
                valleys_indices, _ = find_peaks(
                    -chosen_trace,
                    distance=trace.growth_rate.peaks_identification.minimum_peaks_distance,
                )

                # fill quantities in the section
                trace.growth_rate.peaks_identification.peaks_abscissa = self.results[
                    0
                ].process_time[
                    peaks_indices[
                        trace.growth_rate.peaks_identification.first_peak_index : trace.growth_rate.peaks_identification.last_peak_index
                    ]
                ]
                trace.growth_rate.peaks_identification.valleys_abscissa = self.results[
                    0
                ].process_time[
                    valleys_indices[
                        trace.growth_rate.peaks_identification.first_valley_index : trace.growth_rate.peaks_identification.last_valley_index
                    ]
                ]
                trace.growth_rate.peaks_identification.peaks_ordinate = trace.smoothed_intensity[
                    peaks_indices[
                        trace.growth_rate.peaks_identification.first_peak_index : trace.growth_rate.peaks_identification.last_peak_index
                    ]
                ]
                trace.growth_rate.peaks_identification.valleys_ordinate = trace.smoothed_intensity[
                    valleys_indices[
                        trace.growth_rate.peaks_identification.first_valley_index : trace.growth_rate.peaks_identification.last_valley_index
                    ]
                ]
                # find the peak-to-peak distance
                peak_to_peak = np.diff(
                    trace.growth_rate.peaks_identification.peaks_abscissa.magnitude
                )
                valley_to_valley = np.diff(
                    trace.growth_rate.peaks_identification.valleys_abscissa.magnitude
                )
                # calculate the growth rate
                trace.growth_rate.growth_period = np.mean(
                    np.concatenate([peak_to_peak, valley_to_valley])
                )
                if getattr(trace.growth_rate, "recalculate_on_save") not in [
                    "No",
                    "Yes",
                ]:
                    setattr(trace.growth_rate, "recalculate_on_save", "Yes")
                if (
                    refractive_index is not None
                    and getattr(trace.growth_rate, "recalculate_on_save") == "Yes"
                ):
                    trace.growth_rate.growth_rate = trace.wavelength / (
                        refractive_index * trace.growth_rate.growth_period
                    )

        # plots
        if self.results[0]:
            overview_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=["Reflectance"],
            )

            temperature_object = go.Scattergl(
                x=self.results[0].process_time,
                y=self.results[0].pyrometer_temperature,
                # color=self.results[0].pyrometer_temperature,
                mode="lines+markers",
                line={"width": 2},
                marker={"size": 2},
                showlegend=False,
            )
            overview_fig.add_trace(temperature_object, row=2, col=1)
            for trace in self.results[0].reflectance_wavelengths:
                trace_min = trace.autocorrelation_starting_point
                trace_max = (
                    trace.autocorrelation_starting_point + trace.autocorrelation_period
                )
                single_trace_fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=["Reflectance"],
                )
                go_object = go.Scattergl(
                    x=self.results[0].process_time,
                    y=trace.smoothed_intensity,
                    mode="lines+markers",
                    line={"width": 2},
                    marker={"size": 2},
                    # marker=dict(
                    #     color=np.log10(self.results[0].intensity),
                    #     colorscale="inferno",
                    #     line_width=0,
                    #     showscale=True,
                    # ),
                    name=f"{trace.wavelength.to('nanometer').magnitude:.2f} nm",
                )
                go_object_raw = go.Scattergl(
                    x=self.results[0].process_time,
                    y=trace.raw_intensity,
                    mode="lines+markers",
                    line={"width": 2},
                    marker={"size": 2},
                    # marker=dict(
                    #     color=np.log10(self.results[0].intensity),
                    #     colorscale="inferno",
                    #     line_width=0,
                    #     showscale=True,
                    # ),
                    name=f"{trace.wavelength.to('nanometer').magnitude:.2f} nm",
                )
                positions = np.concatenate(
                    (
                        trace.growth_rate.peaks_identification.peaks_abscissa,
                        trace.growth_rate.peaks_identification.valleys_abscissa,
                    )
                )
                peak_intensities = np.concatenate(
                    (
                        trace.growth_rate.peaks_identification.peaks_ordinate,
                        trace.growth_rate.peaks_identification.valleys_ordinate,
                    )
                )
                go_object_peaks = go.Scattergl(
                    x=positions,
                    y=peak_intensities,
                    mode="markers",
                    marker={"size": 10},
                    name="peaks",
                )
                x_slice = self.results[0].process_time[trace_min:trace_max]
                y_slice = trace.smoothed_intensity[trace_min:trace_max]
                single_trace_fig.add_trace(
                    go_object,
                    row=1,
                    col=1,
                )
                single_trace_fig.add_trace(
                    go_object_raw,
                    row=3,
                    col=1,
                )
                overview_fig.add_trace(
                    go_object_raw,
                    row=1,
                    col=1,
                )
                if (
                    trace.autocorrelated_intensity is not None
                    and trace.autocorrelated_intensity.any()
                ):
                    x_slice = self.results[0].process_time[trace_min:trace_max]
                    y_slice = trace.autocorrelated_intensity[1:]
                    go_object = go.Scattergl(
                        x=x_slice,
                        y=y_slice,
                        mode="lines+markers",
                        line={"width": 2},
                        marker={"size": 2},
                        name=f"Autocorr. {trace.wavelength.magnitude:.2f} nm",
                    )
                    single_trace_fig.add_trace(
                        go_object,
                        row=2,
                        col=1,
                    )
                if getattr(trace.growth_rate, "reflectance_trace") == "Raw":
                    plot_row = 3
                if getattr(trace.growth_rate, "reflectance_trace") == "Smoothed":
                    plot_row = 1
                if getattr(trace.growth_rate, "reflectance_trace") == "Autocorrelated":
                    plot_row = 2
                single_trace_fig.add_trace(
                    go_object_peaks,
                    row=plot_row,
                    col=1,
                )
                single_trace_fig.update_layout(
                    height=800,
                    # width=1000,
                    showlegend=False,
                    dragmode="pan",
                )
                single_trace_fig.update_xaxes(
                    title_text="",
                    autorange=False,
                    range=[trace_min, trace_max],
                    fixedrange=False,
                    ticks="",  # "outside",
                    showticklabels=False,
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    row=1,
                    col=1,
                )
                single_trace_fig.update_xaxes(
                    title_text="",
                    autorange=False,
                    range=[trace_min, trace_max],
                    fixedrange=False,
                    ticks="",  # "outside",
                    showticklabels=True,
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    row=2,
                    col=1,
                )
                single_trace_fig.update_xaxes(
                    title_text="Time [s]",
                    autorange=False,
                    range=[trace_min, trace_max],
                    fixedrange=False,
                    ticks="",  # "outside",
                    showticklabels=True,
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    row=3,
                    col=1,
                )
                single_trace_fig.update_yaxes(
                    title_text="Smoothed Int. [a. u.]",
                    fixedrange=False,
                    ticks="",  # "outside",
                    showticklabels=True,
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    row=1,
                    col=1,
                )
                single_trace_fig.update_yaxes(
                    title_text="Autocorrelated Int. [a. u.]",
                    fixedrange=False,
                    ticks="",  # "outside",
                    showticklabels=True,
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    row=2,
                    col=1,
                )
                single_trace_fig.update_yaxes(
                    title_text="Raw Int. [a. u.]",
                    fixedrange=False,
                    ticks="",  # "outside",
                    showticklabels=True,
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                    mirror=True,
                    row=3,
                    col=1,
                )
                single_trace_fig_json = single_trace_fig.to_plotly_json()
                single_trace_fig_json["config"] = {
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "responsive": False,
                    "displaylogo": False,
                }
                trace.figures = [
                    PlotlyFigure(
                        label=f"{trace.wavelength.to('nanometer').magnitude:.2f} nm",
                        index=1,
                        figure=single_trace_fig_json,
                    )
                ]
            overview_fig.update_layout(
                height=800,
                # width=800,
                showlegend=True,
                legend=dict(
                    orientation="h",  # "h",
                    bgcolor="rgba(0,0,0,0)",
                    # yanchor="bottom",
                    # y=1.02,
                    # xanchor="center",
                    # x=1,
                    yanchor="bottom",
                    y=0.51,
                    xanchor="left",
                    x=0.01,
                    itemwidth=30,
                ),
            )
            overview_fig.update_yaxes(
                title_text="Raw Intensity [a.u.]",
                fixedrange=True,
                ticks="",  # "outside",
                showticklabels=True,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                row=1,
                col=1,
            )
            overview_fig.update_yaxes(
                title_text="Temperature [°C]",
                fixedrange=True,
                ticks="",  # "outside",
                showticklabels=True,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                row=2,
                col=1,
            )
            overview_fig.update_xaxes(
                title_text="",
                # autorange=False,
                # range=[trace_min, trace_max],
                fixedrange=True,
                ticks="",  # "outside",
                showticklabels=False,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                row=1,
                col=1,
            )
            overview_fig.update_xaxes(
                title_text="Time [s]",
                # autorange=False,
                # range=[trace_min, trace_max],
                fixedrange=True,
                ticks="",  # "outside",
                showticklabels=True,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                row=2,
                col=1,
            )
            # overview_fig.update_yaxes(range=[0, 175], autorange=False)
            overview_fig_json = overview_fig.to_plotly_json()
            overview_fig_json["config"] = {
                "scrollZoom": False,
                "responsive": False,
                "displaylogo": False,
                "staticPlot": True,
                "dragmode": False,
            }
            self.figures = [PlotlyFigure(label="figure 1", figure=overview_fig_json)]


#            if self.process.reference:
# with archive.m_context.raw_file(self.process.reference, 'r') as process_file:
#     process_dict = yaml.safe_load(process_file)
#     updated_dep_control['data']['grown_sample'] = GrownSamples(
#             reference=f"../uploads/{archive.m_context.upload_id}/archive/{hash(archive.metadata.upload_id, sample_filename)}#data",
#         ).m_to_dict()

m_package.__init_metainfo__()
