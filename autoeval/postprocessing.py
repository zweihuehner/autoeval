""" This scipt contains postprocessing functions to evaluate the autcalibration model quality """

from .io import Configuration, QuantileInput, ComparisonInputs, InspectionInputs, Input
from .plots import ComparisonScatter, ComparisonTimeseries, InspectionTimeseries, plot_overview
from .office import PowerPointCreator, create_excel_table_from_df

class PostProcessor:
    """
    A class to store, manage and postprocess data from mike simulations.
    """

    def __init__(self, 
                 config: Configuration):
        """
        Initializes a PostProcessor object.

        Args:
            configuration (Configuration): The configuration object.
        """
        self.inputs = []
        self.comparison_inputs = {}
        self.inspection_inputs = {}
        self.config = config

    def add_input(self, model_file: str, model_item: int, name: str, x: float, y: float, 
                  time_interval: list[str, str], observation_file: str | None = None, 
                  observation_item: int | None = None,
                  model_quantiles_input: QuantileInput | None = None, 
                  observation_quantiles_input: QuantileInput | None = None) -> None:
        """
        Adds an InputData object to the PostProcessor.

        Args:
            model_file (str): The path to the model .dfs0 file.
            model_item (int): The item number in the model file.
            name (str): The name of the observed location.
            x (float): The x-coordinate of the observed location.
            y (float): The y-coordinate of the observed location.
            time_interval (list[str, str]): The start and end times of the evaluation period.
            observation_file (str | None, optional): The path to the .dfs0 observation file. Defaults to None.
            observation_item (int | None, optional): The item number in the observation file. Defaults to None.
            model_quantiles_input (QuantileInput | None , optional): The quantile input data for the model. Defaults to None.
            observation_quantiles_input (QuantileInput | None , optional): The quantile input data for the observation. Defaults to None.
        """
        new_input = Input(observation_file=observation_file, observation_item=observation_item, 
                              model_file=model_file, model_item=model_item, name=name, x=x, y=y, 
                              model_quantiles_input=model_quantiles_input, observation_quantiles_input=observation_quantiles_input, 
                              time_interval = time_interval)
        
        if new_input.evaluation_type == "inspection":
            if new_input.quantity not in self.inspection_inputs.keys():
                self.inspection_inputs[new_input.quantity] = InspectionInputs()
            self.inspection_inputs[new_input.quantity].add_input(new_input)
        
        if new_input.evaluation_type == "comparison":
            if new_input.quantity not in self.comparison_inputs.keys():
                self.comparison_inputs[new_input.quantity] = ComparisonInputs()
            self.comparison_inputs[new_input.quantity].add_input(new_input)

        self.inputs.append(new_input)

    def process(self):
        """
        Postprocesses the added inputs.

        The postprocessing includes:
        - creating an overview plot of all stations
        - creating a comparison plot for each quantity with multiple stations (scatter, timeseries, skill tables)
        - creating an inspection plot for each quantity with multiple stations (timeseries)
        - generating a PowerPoint presentation with the plots
        """
        ## Overview
        file_name_overview = plot_overview(input=self.inputs,
                                           folder_out=self.config.output_folder, 
                                           mesh_file=self.config.file_mesh, 
                                           language = self.config.language)

        ## Initialize PowerPoint
        pptx = PowerPointCreator(title = self.config.title_pptx, 
                                 save_file=self.config.pptx_save_path,
                                 base_presentation=self.config.pptx_base_file)
        pptx.add_overview_slide(file_overview = file_name_overview, 
                                title = self.config.title_pptx_overview)

        ## Comparison (Scatter, timeseries and skill tables)
        for i, comparison_input in enumerate(list(self.comparison_inputs.values())):
            
            files_scatter = ComparisonScatter(comparison_input, 
                                                folder_out=self.config.output_folder, 
                                                language = self.config.language).plot()
            
            files_timeseries = ComparisonTimeseries(comparison_input, 
                                                    folder_out=self.config.output_folder, 
                                                    language = self.config.language).plot()
            
            file_table = create_excel_table_from_df(comparison_input,
                                                    folder_out=self.config.output_folder)
            
            for file_scatter, file_timeseries in zip(files_scatter, files_timeseries):
                pptx.add_comparison_slide(file_scatter = file_scatter, 
                                          file_timeseries = file_timeseries, 
                                          title = self.config.title_pptx_comparison)

        ## Inspection (Timeseries)
        for i, inspection_input in enumerate(list(self.inspection_inputs.values())):
            files_timeseries = InspectionTimeseries(inspection_input, 
                                                    folder_out=self.config.output_folder, 
                                                    language = self.config.language).plot()
            
            for file_timeseries in files_timeseries:
                pptx.add_inspection_slide(file_timeseries = file_timeseries)

        pptx.create()