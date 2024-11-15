from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import mikeio
import modelskill as ms
import numpy as np

from autoeval import ROOT_PATH

@dataclass
class Configuration:
    identifier: str
    file_mesh: str

    language: str = 'en'
    output_folder: Optional[Path | str] = field(default=None)
    pptx_save_path: Optional[Path | str] = field(default=None)
    pptx_base_file: Optional[Path | str] = field(default=None)
    title_pptx: Optional[str] = field(default=None)
    title_pptx_overview: Optional[str] = field(default=None)
    title_pptx_comparison: Optional[str] = field(default=None)
    title_pptx_inspection: Optional[str] = field(default=None)

    allowed_languages = ['en', 'de']

    def __post_init__(self):
        """
        Post-init method that is called after the instance is created.
        
        - Checks if the chosen language is in the allowed languages.
        - Calls the _retrieve_automatic_variables method to set variables that are not set manually.
        """
        if self.language not in self.allowed_languages:
            raise ValueError(f"Language must be one of {self.allowed_languages}")
        
        self._retrieve_automatic_variables()

    def _retrieve_automatic_variables(self):
        """
        Retrieves variables that are set automatically if not provided manually.
        
        If the following variables are not set manually, they are set to the following default values:
            - output_folder: "output/<identifier>"
            - title_pptx: "Model Evaluation"
            - title_pptx_overview: "Overview"
            - title_pptx_comparison: "<identifier>"
            - title_pptx_inspection: "<identifier>"
            - pptx_save_path: "output/<identifier>/model_evaluation_<identifier>.pptx"
            - pptx_base_path: "/teamspace/studios/this_studio/core/PowerPoint_Base.pptx"
        """
        if self.output_folder is None:
            self.output_folder = Path("output") / self.identifier.lower().replace(' ', '')
            self.output_folder.mkdir(parents=True, exist_ok=True)
        if self.title_pptx is None:
            self.title_pptx = "Model Evaluation"
        if self.title_pptx_overview is None:
            self.title_pptx_overview = "Overview"
        if self.title_pptx_comparison is None:
            self.title_pptx_comparison = f"{self.identifier}"
        if self.title_pptx_inspection is None:
            self.title_pptx_inspection = f"{self.identifier}"
        if self.pptx_save_path is None:
            self.pptx_save_path = Path(self.output_folder) / f"model_evaluation_{self.identifier.lower().replace(' ', '')}.pptx"
        if self.pptx_base_file is None:
            self.pptx_base_file = ROOT_PATH / "basic" / "PowerPoint_Base.pptx"
            if not self.pptx_base_file.exists():
                raise FileNotFoundError(f"PowerPoint_Base.pptx not found in {self.pptx_base_file}")

@dataclass
class QuantileInput:
    item_1: int
    item_2: int
    label_1: int
    label_2: int

@dataclass
class QuantileRange:
    quantile_1: pd.Series | pd.DataFrame | None = None
    quantile_2: pd.Series | pd.DataFrame | None = None
    label_1: str | None = None
    label_2: str | None = None

@dataclass
class Input:
    model_file: str
    model_item: int
    name: str
    x: float
    y: float
    time_interval: list[str, str]

    model_quantiles_data: QuantileRange = QuantileRange()
    observation_quantiles_data: QuantileRange = QuantileRange()
    observation_file: str | None = None
    observation_item: int | None = None
    model_quantiles_input: QuantileInput | None = None
    observation_quantiles_input: QuantileInput | None = None

    quantity: str = field(init=False)
    unit: str = field(init=False)
    evaluation_type: str = field(init=False)
    max_val: float = field(init=False)
    min_val: float = field(init=False)
    max_dev: Optional[float] = field(init=False, default = None)
    min_dev: Optional[float] = field(init=False, default = None)

    def __post_init__(self):
        """
        Post initialization method to set input type based on presence of observation file and item.
        Sets input type to either "inspection" or "comparison".
        If input type is "inspection", sets model data and gets min and max values. If model quantiles are given, sets them.
        If input type is "comparison", sets model and observation data, gets min and max values and min and max deviations.
        If model or observation quantiles are given, sets them.
        """
        self._get_input_type()
        if self.evaluation_type == "inspection":
            dat_model = mikeio.read(self.model_file, time = slice(self.time_interval[0], self.time_interval[1]))
            model_data_tmp = ms.PointModelResult(dat_model, item = self.model_item, name =f"model {self.name}", x=self.x, y=self.y)
            self.quantity = model_data_tmp.quantity.name
            self.unit = self._parse_unit(model_data_tmp.quantity.unit)
            self.model_data = model_data_tmp.to_dataframe()

            if self.model_quantiles_input is not None:
                self.model_quantiles_data = self._get_quantile_data(dat_model, self.model_quantiles_input)
            self.min_val, self.max_val = self._get_min_max_vals([model_data_tmp, 
                                                                 self.model_quantiles_data.quantile_1, 
                                                                 self.model_quantiles_data.quantile_2])

            if self.model_quantiles_input is None:
                self.model_quantiles_data = None

        elif self.evaluation_type == "comparison":
            dat_model = mikeio.read(self.model_file, time = slice(self.time_interval[0], self.time_interval[1]))
            model_data_tmp = ms.PointModelResult(dat_model, item=self.model_item, name =f"model {self.name}", x=self.x, y=self.y)
            dat_observation = mikeio.read(self.observation_file, time = slice(self.time_interval[0], self.time_interval[1]))
            observation_data_tmp = ms.PointObservation(dat_observation, item=self.observation_item, name =f"observation {self.name}", x=self.x, y=self.y)
            self.c = ms.match(obs=observation_data_tmp, mod=model_data_tmp).sel(start=self.time_interval[0], end=self.time_interval[1])
            self.observation_data = self.c.to_dataframe().iloc[:, 0]
            self.model_data = self.c.to_dataframe().iloc[:, 1]
            self.quantity = model_data_tmp.quantity.name
            self.unit = self._parse_unit(model_data_tmp.quantity.unit)

            if self.model_quantiles_input is not None:
                self.model_quantiles_data = self._get_quantile_data(dat_model, self.model_quantiles_input)

            if self.observation_quantiles_input is not None:
                self.observation_quantiles_data = self._get_quantile_data(dat_observation, self.observation_quantiles_input)

            self.min_val, self.max_val = self._get_min_max_vals([self.model_data, self.observation_data, 
                                                                 self.model_quantiles_data.quantile_1, self.model_quantiles_data.quantile_2,
                                                                 self.observation_quantiles_data.quantile_1, self.observation_quantiles_data.quantile_2])
            self.min_dev, self.max_dev = self._get_min_max_devs(self.model_data, self.observation_data)

            if self.model_quantiles_input is None:
                self.model_quantiles_data = None
            if self.observation_quantiles_input is None:
                self.observation_quantiles_data = None

    def _get_input_type(self) -> None:
        """
        Determines the type of input data (either inspection or comparison) based on whether an observation file is provided.
        """
        if self.observation_file is None:
            self.evaluation_type = "inspection"
        elif self.observation_file is not None:
            self.evaluation_type = "comparison"
    
    def _get_min_max_vals(self, data: list[pd.Series | None]) -> None:
        """
        Returns the minimum and maximum values from a list of pandas Series or None values.
        Excludes None values from the calculation.
        """
        data = [x.values for x in data if x is not None]
        return np.min(np.concatenate(data)), np.max(np.concatenate(data))

    def _get_min_max_devs(self, model_data: pd.Series, observation_data: pd.Series) -> None:
        """
        Calculates the minimum and maximum difference between model and observation data.

        Args:
            model_data (pd.Series): The model data.
            observation_data (pd.Series): The observation data.

        Returns:
            tuple: The minimum and maximum difference between the model and observation data.
        """
        dev = model_data - observation_data
        return np.min(dev), np.max(dev)

    def _get_quantile_data(self, data, quantile_input: QuantileInput) -> None:
        """
        Extracts two quantile series from the given data based on the item numbers and labels in the QuantileInput object.
        The resulting quantile series are returned as a QuantileRange object.
        """
        q1_data_tmp = ms.PointModelResult(data, item = quantile_input.item_1, name = quantile_input.label_1, x=self.x, y=self.y)
        q2_data_tmp = ms.PointModelResult(data, item = quantile_input.item_2, name = quantile_input.label_2, x=self.x, y=self.y)
        return QuantileRange(quantile_1=q1_data_tmp.to_dataframe().iloc[:, 0], 
                            quantile_2=q2_data_tmp.to_dataframe().iloc[:, 0], 
                            label_1=quantile_input.label_1, 
                            label_2=quantile_input.label_2)


    def _parse_unit(self, unit) -> str:
        """
        Converts unit notation from "m^3/s" to "m³/s".

        Args:
            unit (str): The unit string to be converted.

        Returns:
            str: The converted unit string.
        """
        if unit == "m^3/s":
            return "m³/s"
        else:
            return unit

@dataclass
class InspectionInputs:
    inputs: list[Input] = field(default_factory=list)
    max_val: float = None
    min_val: float = None

    quantity: str = field(init=False, default = None)
    unit: str = field(init=False, default = None)

    def add_input(self, input: Input):
        """
        Adds an InputData object to the InspectionInputs.

        Args:
            input (Input): The InputData object to be added.

        Raises:
            ValueError: If the input is not of evaluation_type 'inspection'.
        """
        if input.evaluation_type != 'inspection':
            raise ValueError("Input must be of evaluation_type 'inspection'")
        
        self._check_sanity(input)
        self._update_min_max(input.min_val, input.max_val)
        
        self.inputs.append(input)

    def _check_sanity(self, input):
        """
        Checks that the quantity and unit of the given input is the same as the ones
        already stored in this InspectionInputs instance. If not, raises a ValueError.

        Args:
            input (Input): The input to be checked.

        Raises:
            ValueError: If the quantity or unit of the given input is not the same as the ones
                stored in this InspectionInputs instance.
        """
        if self.quantity is None:
            self.quantity = input.quantity
            self.unit = input.unit
        else:
            if self.quantity != input.quantity:
                raise ValueError(f"This Inputs Instance only saves inputs of one quantity {self.quantity}, but got {input.quantity})")
            if self.unit != input.unit:
                raise ValueError(f"This Inputs Instance only saves inputs of one unit {self.unit}, but got {input.unit})")

    def _update_min_max(self, min_val, max_val):
        """
        Updates the minimum and maximum values of the inputs stored in this InspectionInputs instance.

        The minimum and maximum values are updated only if the current minimum and maximum values are None, or if the
        given minimum and maximum values are respectively smaller and larger than the current minimum and maximum values.

        Args:
            min_val (float): The minimum value of the new input.
            max_val (float): The maximum value of the new input.
        """
        if self.min_val is None:
            self.min_val = min_val
        else:
            if self.min_val > min_val:
                self.min_val = min_val

        if self.max_val is None:
            self.max_val = max_val
        else: 
            if self.max_val < max_val:
                self.max_val = max_val

@dataclass
class ComparisonInputs:
    inputs: list[Input] = field(default_factory=list)
    max_val: float = None
    min_val: float = None
    max_dev: float = None
    min_dev: float = None

    quantity: str = field(init=False, default = None)
    unit: str = field(init=False, default = None)
    cc: ms.ComparerCollection = field(init=False, default = None)
    skill: pd.DataFrame = field(init=False, default = None)

    def add_input(self, input: Input):
        """
        Adds an Input object to the ComparisonInputs.

        This method ensures that the input is of type 'comparison' and updates the
        ComparerCollection, min/max values, and performs a sanity check before adding
        the input to the list.

        Args:
            input (Input): The Input object to be added. Must be of evaluation_type 'comparison'.

        Raises:
        
            ValueError: If the input is not of evaluation_type 'comparison'.
        """
        if input.evaluation_type != 'comparison':
            raise ValueError("Input must be of evaluation_type 'comparison'")
        
        self._update_cc_skill(input)
        self._update_min_max(input.min_val, input.max_val, input.min_dev, input.max_dev)
        self._check_sanity(input)
        self.inputs.append(input)

    def _update_cc_skill(self, input):
        """
        Updates the ComparerCollection and associated skill DataFrame with a new input.

        Args:
            input (Input): The Input object to be added.

        Notes:
            The index of the skill DataFrame is set to 'Station' and the index values are stripped of 'observation '
            The columns 'n', 'x', and 'y' are dropped from the skill DataFrame if they exist
            The skill DataFrame is rounded to 3 decimal places
        """
        if self.cc is None:
            self.cc = ms.ComparerCollection(input.c)
        else:
            self.cc += input.c

        self.skill = self.cc.skill()

        names = [name for name in self.skill.index.names if name != 'observation']
        self.skill = self.skill.reset_index(level=names)
        self.skill = self.skill.drop(names, axis=1)

        self.skill.index.names = ['Station']
        self.skill.index = self.skill.index.str.lstrip('observation ')
        self.skill = self.skill.drop(columns = ["n", "x", "y"], errors = "ignore").round(3)

    def _update_min_max(self, min_val, max_val, min_dev, max_dev):
        """
        Updates the minimum and maximum values of the inputs stored in this ComparisonInputs instance.

        The minimum and maximum values are updated only if the current minimum and maximum values are None, or if the
        given minimum and maximum values are respectively smaller and larger than the current minimum and maximum values.

        Args:
            min_val (float): The minimum value of the new input.
            max_val (float): The maximum value of the new input.
            min_dev (float): The minimum deviation of the new input.
            max_dev (float): The maximum deviation of the new input.
        """
        if self.min_val is None:
            self.min_val = min_val
        else:
            if self.min_val > min_val:
                self.min_val = min_val

        if self.min_dev is None:
            self.min_dev = min_dev
        else:
            if self.min_dev > min_dev:
                self.min_dev = min_dev

        if self.max_val is None:
            self.max_val = max_val
        else: 
            if self.max_val < max_val:
                self.max_val = max_val

        if self.max_dev is None:
            self.max_dev = max_dev
        else: 
            if self.max_dev < max_dev:
                self.max_dev = max_dev

    def _check_sanity(self, input):
        """
        Checks that the quantity and unit of the given input is the same as the ones
        already stored in this ComparisonInputs instance. If not, raises a ValueError.

        Args:
            input (Input): The input to be checked.

        Raises:
            ValueError: If the quantity or unit of the given input is not the same as the ones
                stored in this ComparisonInputs instance.
        """
        if self.quantity is None:
            self.quantity = input.quantity
            self.unit = input.unit
        else:
            if self.quantity != input.quantity:
                raise ValueError(f"This Inputs Instance only saves inputs of one quantity {self.quantity}, but got {input.quantity})")
            if self.unit != input.unit:
                raise ValueError(f"This Inputs Instance only saves inputs of one unit {self.unit}, but got {input.unit})")
        