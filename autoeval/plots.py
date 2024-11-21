import mikeio
import modelskill as ms
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import contextily as ctx

from .io import ComparisonInputs, InspectionInputs, Input, QuantileRange

class ComparisonScatter:

    plot_identifier: str = "ComparisonScatter"

    def __init__(self, data: ComparisonInputs, folder_out: str, title: str = "", 
                xlabel: str | None = None, ylabel: str | None = None, 
                unit: str | None = None, quantity: str | None = None, 
                language: str = "en"):
        """
        Initializes the ComparisonScatter class.

        Args:
            data (ComparisonInputs): The comparison input data.
            folder_out (str): The path to save the scatter plot to.
            title (str): The title text for the scatter plot. Defaults to "".
            xlabel (str): The label for the x-axis. Defaults to None.
            ylabel (str): The label for the y-axis. Defaults to None.
            unit (str): The unit to use for the plot. Defaults to None.
            quantity (str): The quantity to use for the plot. Defaults to None.
            language (str): The language to use for the plot. Defaults to "en".
        """
        allowed_languages = ["de", "en"]
        if language not in allowed_languages:
            raise ValueError(f"Language must be one of {allowed_languages}")
        
        self.title = title
        self.xlabel = xlabel 
        self.ylabel = ylabel
        self.unit = unit
        self.data = data
        self.quantity = quantity

        self.folder_out = folder_out

        self._description(language = language)

    def plot(self):
        """
        Plots a scatter plot for each comparer in the comparer collection.

        Args:
            None

        Returns:
            list: A list of file names for the scatter plots.
        """
        file_names = [] 

        for c in self.data.cc:

            file_names.append(self.plot_comparison_scatter(c=c, 
                                                      title=self.title, 
                                                      xlabel=self.xlabel, 
                                                      ylabel=self.ylabel, 
                                                      quantity=self.quantity,
                                                      plot_identifier=self.plot_identifier,
                                                      folder_out=self.folder_out))

        return file_names
    
    @staticmethod
    def plot_comparison_scatter(c: ms.ComparerCollection, 
                            title: str, 
                            xlabel:str, 
                            ylabel:str, 
                            quantity: str, 
                            plot_identifier: str,
                            folder_out:str):
        """
        Plots a scatter plot for a comparer collection.

        Args:
            c (ms.ComparerCollection): The comparer collection to plot.
            title (str): The title text for the scatter plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            quantity (str): The quantity to use for the plot.
            plot_identifier (str): Part of the output filename.
            folder_out (str): The path to save the scatter plot to.

        Returns:
            str: The file name of the saved scatter plot.
        """
        name = c.name.split(" ")[1]

        c.plot.scatter(
            skill_table=True, 
            show_points=True, 
            cmap='RdYlGn', 
            title=f'{title} {name}',
            xlabel=xlabel, 
            ylabel=ylabel,
            figsize=(10, 7)
        )

        Path(folder_out).mkdir(parents=True, exist_ok=True)
        file_out = Path(folder_out) / f"{plot_identifier}_{quantity.replace(' ', '')}_{name}.png"
        plt.savefig(file_out, dpi=300, bbox_inches='tight')
        
        return file_out
    
    def _description(self, language:str):
        """
        Sets the labels, unit, and quantity name based on the language.

        If the labels, unit, or quantity name are not set, this method will
        set them based on the language and the quantity.

        Args:
            language (str): The language to use. Must be either "en" or "de".

        Raises:
            ValueError: If the quantity is not supported yet.
        """
        if self.unit is None:
            self.unit = self.data.unit

        if self.quantity is None:
            self.quantity = self.data.quantity
        
        if language == "en":
            self.model_label = "Model"
            self.observation_label = "Observation"
        elif language == "de":
            self.model_label = "Modell"
            self.observation_label = "Messung"

        if "Water Level" in self.data.quantity or "Surface Elevation" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Surface Elevation"
            elif language == "de":
                self.quantity_name = "Wasserstand"
        elif "Discharge" in self.data.quantity or "Volume Flux" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Discharge"
            elif language == "de":
                self.quantity_name = "Abfluss"
        elif "Current Speed" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Current Speed"
            elif language == "de":
                self.quantity_name = "Fließgeschwindigkeit"
        else:
            raise ValueError(f"{self.quantity} not supported yet. Please add it to the ComparisonScatter class.")

        if self.xlabel is None:
            self.xlabel = f"{self.observation_label} - {self.quantity_name} [{self.unit}]" 

        if self.ylabel is None:
            self.ylabel = f"{self.model_label} - {self.quantity_name} [{self.unit}]" 

class ComparisonTimeseries:

    plot_identifier: str = "ComparisonTimeseries"

    def __init__(self, data: ComparisonInputs, folder_out: str, title: str = "", 
                xlabel: str | None = None, ylabel: str | None = None, 
                y2label: str | None = None,
                unit: str | None = None, quantity: str | None = None, 
                language: str = "en"):
        """
        Initializes a ComparisonTimeseries object.

        Args:
            data (ComparisonInputs): The comparison input data.
            folder_out (str): The path to save the timeseries plot to.
            title (str, optional): The title text for the timeseries plot. Defaults to "".
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            y2label (str | None, optional): The label for the secondary y-axis. Defaults to None.
            unit (str | None, optional): The unit to use for the plot. Defaults to None.
            quantity (str | None, optional): The quantity to use for the plot. Defaults to None.
            language (str, optional): The language to use for the plot. Must be either "en" or "de". Defaults to "en".

        Raises:
            ValueError: If the language is not one of the allowed languages.
        """
        allowed_languages = ["de", "en"]
        if language not in allowed_languages:
            raise ValueError(f"Language must be one of {allowed_languages}")
        
        self.title = title
        self.xlabel = xlabel 
        self.ylabel = ylabel
        self.y2label = y2label
        self.unit = unit
        self.data = data
        self.quantity = quantity

        self.folder_out = folder_out

        self._description(language = language)

    def plot(self):
        """
        Plots a comparison timeseries plot for each input in the data.

        Args:
            None

        Returns:
            list[str]: A list of file names of the saved plots.
        """
        file_names = [] 

        y1_dif = np.abs(self.data.min_val - self.data.max_val)
        y2_dif = np.abs(self.data.min_dev - self.data.max_dev)
    
        y1_lim = [self.data.min_val - 0.1 * y1_dif, self.data.max_val + 0.1 * y1_dif]
        y2_lim = [self.data.min_dev - 0.1 * y2_dif, self.data.max_dev + 0.1 * y2_dif]

        for input in self.data.inputs:

            file_names.append(self.plot_comparison_timeseries(name = input.name,
                                                              mod = input.model_data,
                                                              mod_label = self.model_label,
                                                              obs = input.observation_data, 
                                                              obs_label = self.observation_label, 
                                                              dev = input.model_data - input.observation_data, 
                                                              title = self.title, 
                                                              xlabel = self.xlabel, 
                                                              ylabel = self.ylabel, 
                                                              y2label = self.y2label,
                                                              quantity = self.quantity,
                                                              plot_identifier=self.plot_identifier,
                                                              folder_out= self.folder_out,
                                                              y1_lim = y1_lim,
                                                              y2_lim = y2_lim,
                                                              model_quantiles = input.model_quantiles_data,
                                                              observation_quantiles = input.observation_quantiles_data
                                                              ))
        
        return file_names
    
    @staticmethod
    def plot_comparison_timeseries(name: str, 
                                mod: pd.Series, 
                                mod_label: str, 
                                obs: pd.Series, 
                                obs_label: str,
                                dev: pd.Series, 
                                title: str, 
                                xlabel: str, 
                                ylabel: str, 
                                y2label: str,
                                folder_out: str, 
                                quantity: str,
                                plot_identifier: str,
                                y1_lim: list[float | None, float | None], 
                                y2_lim: list[float | None, float | None], 
                                model_quantiles: QuantileRange | None = None,
                                observation_quantiles: QuantileRange | None = None):
        """
        Plots a comparison timeseries plot with model, observation, and deviation data.

        This method creates a two-panel plot: the upper panel displays the model and observation
        timeseries, while the lower panel shows the deviation between them. It also supports
        quantile shading for both model and observation data.

        Args:
            name (str): The name identifier for the plot file.
            mod (pd.Series): The model data timeseries.
            mod_label (str): The label for the model data.
            obs (pd.Series): The observation data timeseries.
            obs_label (str): The label for the observation data.
            dev (pd.Series): The deviation timeseries (model - observation).
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis of the upper panel.
            y2label (str): The label for the y-axis of the lower panel.
            folder_out (str): The directory to save the plot file.
            quantity (str): The quantity being plotted.
            plot_identifier (str): Part of the output filename.
            y1_lim (list[float | None, float | None]): The y-axis limits for the upper panel.
            y2_lim (list[float | None, float | None]): The y-axis limits for the lower panel.
            model_quantiles (QuantileRange | None, optional): The quantile range for the model data, if available.
            observation_quantiles (QuantileRange | None, optional): The quantile range for the observation data, if available.

        Returns:
            str: The file path of the saved plot image.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.7, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        linewidth = 2

        if model_quantiles is not None:
            ax1.fill_between(model_quantiles.quantile_1.index, model_quantiles.quantile_1, model_quantiles.quantile_2, 
                             zorder=-1, color="red", alpha=0.1, label=f"{mod_label} ({model_quantiles.label_1}-{model_quantiles.label_2})")
            mod_label = f"{mod_label} (mean)"
            ax1.legend([f""])
        if observation_quantiles is not None:
            ax1.fill_between(observation_quantiles.quantile_1.index, observation_quantiles.quantile_1, observation_quantiles.quantile_2, 
                             zorder=-1, color="black", alpha=0.1, label=f"{obs_label} ({observation_quantiles.label_1}-{observation_quantiles.label_2})")
            obs_label = f"{obs_label} (mean)"
            ax1.legend([f""])

        ax1.plot(mod.index, mod, color="r", linewidth=linewidth, label=mod_label)
        ax1.plot(obs.index, obs, color="k", linewidth=linewidth, label=obs_label)

        ax1.set_ylabel(ylabel, color='k')
        ax1.set_title(f"{title} {name}")
        ax1.grid(axis='y', linestyle='--', linewidth=0.5)
        ax1.set_ylim([y1_lim[0], y1_lim[1]])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
        ax1.set_xlim(mod.index.min(), mod.index.max())
        ax1.legend(frameon=False)

        ax2.plot(dev.index, dev, color="gray", linestyle="--", linewidth=1)
        ax2.set_ylabel(f'{y2label}', color='k')
        ax2.axhline(y=0, color='lightgray', linestyle='--', linewidth=1)
        ax2.set_ylim([y2_lim[0], y2_lim[1]])
        ax2.set_xlim(mod.index.min(), mod.index.max())
        ax2.set_xlabel(xlabel)
        plt.tight_layout()
        fig.autofmt_xdate()

        Path(folder_out).mkdir(parents=True, exist_ok=True)

        file_out = Path(folder_out) / f"{plot_identifier}_{quantity.replace(' ', '')}_{name}.png"

        plt.savefig(file_out, dpi=300, bbox_inches='tight')

        return file_out
    
    def _description(self, language:str):
        """
        Sets the labels, unit, and quantity name based on the language.

        If the labels, unit, or quantity name are not set, this method will
        set them based on the language and the quantity.

        Args:
            language (str): The language to use. Must be either "en" or "de".

        Raises:
            ValueError: If the quantity is not supported yet.
        """
        if self.unit is None:
            self.unit = self.data.unit

        if self.quantity is None:
            self.quantity = self.data.quantity
        
        if language == "en":
            self.model_label = "Model"
            self.observation_label = "Observation"
            self.dev_label = f"Deviation"
            self.time_label = "Time"
        elif language == "de":
            self.model_label = "Modell"
            self.observation_label = "Messung"
            self.dev_label = f"Abweichung"
            self.time_label = "Zeit"

        if "Water Level" in self.data.quantity or "Surface Elevation" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Surface Elevation"
            elif language == "de":
                self.quantity_name = "Wasserstand"
        elif "Discharge" in self.data.quantity or "Volume Flux" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Discharge"
            elif language == "de":
                self.quantity_name = "Abfluss"
        elif "Current Speed" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Current Speed"
            elif language == "de":
                self.quantity_name = "Fließgeschwindigkeit"
        else:
            raise ValueError(f"{self.quantity} not supported yet. Please add it to the ComparisonScatter class.")

        if self.xlabel is None:
            self.xlabel = f"{self.time_label}"

        if self.ylabel is None:
            self.ylabel = f"{self.quantity_name} [{self.unit}]"  

        if self.y2label is None:
            self.y2label = f"{self.dev_label} [{self.unit}]"  
            

class InspectionTimeseries:

    plot_identifier: str = "InspectionTimeseries"

    def __init__(self, data: InspectionInputs, folder_out: str, title: str = "", 
            xlabel: str | None = None, ylabel: str | None = None, 
            unit: str | None = None, quantity: str | None = None, 
                language: str = "en"):
        """
        Initializes a InspectionTimeseries object.

        Args:
            data (InspectionInputs): The inspection input data.
            folder_out (str): The path to save the timeseries plot to.
            title (str, optional): The title text for the timeseries plot. Defaults to "".
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            unit (str | None, optional): The unit to use for the plot. Defaults to None.
            quantity (str | None, optional): The quantity to use for the plot. Defaults to None.
            language (str, optional): The language to use for the plot. Must be either "en" or "de". Defaults to "en".

        Raises:
            ValueError: If the language is not one of the allowed languages.
        """
        allowed_languages = ["de", "en"]
        if language not in allowed_languages:
            raise ValueError(f"Language must be one of {allowed_languages}")
        
        self.title = title
        self.xlabel = xlabel 
        self.ylabel = ylabel
        self.unit = unit
        self.data = data
        self.quantity = quantity

        self.folder_out = folder_out

        self._description(language = language)

    def plot(self):
        """
        Plots an inspection timeseries plot for each input in the data.

        This method calculates the y-axis limits based on the minimum and maximum values
        of the data and generates a plot for each input using the `plot_inspection_timeseries` method.

        Returns:
            list[str]: A list of file names of the saved plots.
        """
        file_names = [] 

        y1_dif = np.abs(self.data.min_val - self.data.max_val)
    
        y1_lim = [self.data.min_val - 0.1 * y1_dif, self.data.max_val + 0.1 * y1_dif]

        for input in self.data.inputs:

            file_names.append(self.plot_inspection_timeseries(name = input.name,
                                                              mod = input.model_data,
                                                              mod_label = self.model_label,
                                                              title = self.title, 
                                                              xlabel = self.xlabel, 
                                                              ylabel = self.ylabel, 
                                                              quantity = self.quantity,
                                                              folder_out= self.folder_out,
                                                              y1_lim = y1_lim,
                                                              plot_identifier=self.plot_identifier,
                                                              model_quantiles = input.model_quantiles_data))
        
        return file_names
    
    @staticmethod
    def plot_inspection_timeseries(
                                name: str, 
                                mod: pd.Series, 
                                mod_label: str, 
                                title: str, 
                                xlabel: str, 
                                ylabel: str, 
                                folder_out: str, 
                                quantity: str,
                                plot_identifier: str,
                                y1_lim: list[float | None, float | None], 
                                model_quantiles: QuantileRange | None = None) -> str:
        """
        Plots an inspection timeseries plot with model data and optional quantiles.

        Args:
            name (str): The name identifier for the plot file.
            mod (pd.Series): The model data timeseries.
            mod_label (str): The label for the model data.
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            folder_out (str): The directory to save the plot file.
            quantity (str): The quantity being plotted.
            plot_identifier (str): Part of the output filename.
            y1_lim (list[float | None, float | None]): The y-axis limits for the upper panel.
            model_quantiles (QuantileRange | None, optional): The quantile range for the model data, if available.

        Returns:
            str: The file path of the saved plot image.
        """
        fig, ax1 = plt.subplots(figsize=(11.7, 4.5))

        linewidth = 2

        if model_quantiles is not None:
            ax1.plot(mod.index, mod, color="k", linewidth=linewidth, label=f"{mod_label} (mean)")
            ax1.fill_between(model_quantiles.quantile_1.index, model_quantiles.quantile_1, model_quantiles.quantile_2, 
                             zorder=-1, color="red", alpha=0.1, label=f"{mod_label} ({model_quantiles.label_1}-{model_quantiles.label_2})")           
            ax1.legend([f""])
        else:
            ax1.plot(mod.index, mod, color="k", linewidth=linewidth, label=mod_label)

        ax1.set_ylabel(ylabel, color='k')
        ax1.set_xlabel(xlabel, color='k')
        ax1.set_title(f"{title} {name}")
        ax1.grid(axis='y', linestyle='--', linewidth=0.5)
        ax1.set_ylim([y1_lim[0], y1_lim[1]])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
        ax1.set_xlim(mod.index.min(), mod.index.max())
        ax1.legend(frameon=False)
        fig.autofmt_xdate()

        plt.tight_layout()

        Path(folder_out).mkdir(parents=True, exist_ok=True)

        file_out = Path(folder_out) / f"{plot_identifier}_{quantity.replace(' ', '')}_{name}.png"

        plt.savefig(file_out, dpi=300, bbox_inches='tight')

        return file_out
    
    def _description(self, language:str):
        """
        Sets the labels, unit, and quantity name based on the language.

        If the labels, unit, or quantity name are not set, this method will
        set them based on the language and the quantity.

        Args:
            language (str): The language to use. Must be either "en" or "de".

        Raises:
            ValueError: If the quantity is not supported yet.
        """
        if self.unit is None:
            self.unit = self.data.unit

        if self.quantity is None:
            self.quantity = self.data.quantity
        
        if language == "en":
            self.model_label = "Model"
            self.time_label = "Time"
        elif language == "de":
            self.model_label = "Modell"
            self.time_label = "Zeit"

        if "Water Level" in self.data.quantity or "Surface Elevation" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Surface Elevation"
            elif language == "de":
                self.quantity_name = "Wasserstand"
        elif "Discharge" in self.data.quantity or "Volume Flux" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Discharge"
            elif language == "de":
                self.quantity_name = "Abfluss"
        elif "Current Speed" in self.data.quantity:
            if language == "en":
                self.quantity_name = "Current Speed"
            elif language == "de":
                self.quantity_name = "Fließgeschwindigkeit"
        else:
            raise ValueError(f"{self.quantity} not supported yet. Please add it to the ComparisonScatter class.")

        if self.xlabel is None:
            self.xlabel = f"{self.time_label}"

        if self.ylabel is None:
            self.ylabel = f"{self.quantity_name} [{self.unit}]"  

def plot_overview(input: list[Input], mesh_file: str, crs: str="epsg:25832", zoom: int = 12, center: list[float, float] | None = None, 
                  c_zoom: int = 1000, show_mesh: bool = False, folder_out: str = "output/", language = "en") -> Path:

    """
    Plot the locations of the input data points and the used mesh on a map.

    Args:
        input (list[Input]): The input data to plot.
        mesh_file (str): The path to the mesh file.
        crs (str, optional): The coordinate reference system of the mesh. Defaults to "epsg:25832".
        zoom (int, optional): The zoom level of the map. Defaults to 12.
        center (list[float, float] | None, optional): The center of the map. Defaults to None.
        c_zoom (int, optional): The zoom level of the map for the center area. Defaults to 1000.
        show_mesh (bool, optional): Whether to show the mesh. Defaults to False.
        folder_out (str, optional): The folder to save the plot in. Defaults to "output/".
        language (str, optional): The language of the plot. Defaults to "en".

    Returns:
        str: The path to the saved plot.
    """
    allowed_languages = ["de", "en"]
    if language not in allowed_languages:
        raise ValueError(f"Language must be one of {allowed_languages}")
    
    if language == "de":
        title = "Evaluierungsstandorte"
        bath = "Bathymetrie [m]"
        easting = "Rechtswert [m]"
        northing = "Hochwert [m]"

    elif language == "en":
        title = "Evaluation Locations"
        bath = "Bathymetry [m]"
        easting = "Easting [m]"
        northing = "Northing [m]"

    dfs_mesh = mikeio.Mesh(mesh_file)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    dfs_mesh.plot(show_mesh=show_mesh, cmap="gist_earth", vmin=-22, vmax=4, title=title, label=bath, ax=ax)

    if center is not None:
        ax.set_xlim(center[0] - c_zoom, center[0] + c_zoom)
        ax.set_ylim(center[1] - c_zoom, center[1] + c_zoom)

    for input_i in input:
        ax.plot(input_i.x, input_i.y, 'ow', markersize=5)
        rotation = 45 
        ax.text(input_i.x, input_i.y, input_i.name, fontsize=10, ha='left', va='bottom', color='w', rotation=rotation)

    source = ctx.providers.Esri.WorldImagery
    ctx.add_basemap(ax, crs=crs, source=source, zoom=zoom, alpha=0.8, attribution_size=0)
    
    ax.set_xlabel(easting)
    ax.set_ylabel(northing)

    plt.tight_layout()    

    file_out = Path(folder_out) / "overview.png"

    plt.savefig(file_out, dpi=300, bbox_inches='tight')

    return file_out