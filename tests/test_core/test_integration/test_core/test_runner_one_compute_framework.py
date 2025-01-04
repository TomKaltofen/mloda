from typing import Any, Dict, List, Optional, Set, Type, Union
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.api.request import mlodaAPI
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options


class EngineRunnerTest(AbstractFeatureGroup):
    f_name = "EngineRunnerTest1"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.get_options_key("config") == "test":
            return {cls.f_name: [2, 4, 6]}
        return {cls.f_name: [1, 2, 3]}

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.f_name})


class EngineRunnerTest2(AbstractFeatureGroup):
    f_name = "EngineRunnerTest2"
    f_name2 = "EngineRunnerTest1"

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if cls.f_name in feature_name.name:  # type: ignore
            return True
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.int32_of(self.f_name2)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pc.multiply(data.column(cls.f_name2), 2)

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        if config.get("config"):
            return FeatureName(f"{self.f_name}-{config.get('config')}")
        return FeatureName(self.get_class_name())


class EngineRunnerTest3(AbstractFeatureGroup):
    feature_2 = Feature.int32_of("EngineRunnerTest2")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {self.feature_2}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pc.multiply(data.column(cls.feature_2.get_name()), 3)


class EngineRunnerTest4(AbstractFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data = {
            f"{cls.get_class_name()}_0": [1.1, 2.2, 3.3],
            f"{cls.get_class_name()}_2": [4.4, 5.5, 6.6],
            f"{cls.get_class_name()}_3": [7.7, 8.8, 9.9],
            f"{cls.get_class_name()}_1": [10.1, 11.2, 12.3],
        }

        arrays = [pa.array(data[col]) for col in data if col in features.get_all_names()]
        schema = pa.schema([(col, pa.float64()) for col in data if col in features.get_all_names()])
        return pa.Table.from_arrays(arrays, schema=schema)

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({f"{cls.get_class_name()}_{cnt}" for cnt in range(4)})


class SumFeature(AbstractFeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if "sum_of_" in feature_name.name:  # type: ignore
            return True
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_names = options.get("sum")
        return {Feature.int32_of(value) for value in set(feature_names)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cols = list(features.get_options_key("sum"))
        new = []
        for col in cols:
            if col not in data.schema.names:
                new.append(col.split("_", 1)[1])
            else:
                new.append(col)
        return pa.array([pc.sum([pc.sum(data.column(col)) for col in new])] * len(data.column(new[0])))

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        ending = "".join(config.get("sum"))
        return FeatureName(f"{self.get_class_name()}_{ending}")


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        # ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestEngineRunnerOneComputeFramework:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def basic_runner(
        self,
        features: Features,
        parallelization_modes: Set[ParallelizationModes],
        flight_server: Any,
        function_extender: Any = None,
    ) -> List[Any]:
        compute_framework: Set[Type[ComputeFrameWork]] = {PyarrowTable}

        results = mlodaAPI.run_all(
            features, compute_framework, None, None, parallelization_modes, flight_server, function_extender
        )
        # assert_flight_infos(flight_server)
        return results

    def test_runner_cfw_single_feature(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        features = self.get_features(["EngineRunnerTest1"])
        result = self.basic_runner(features, modes, flight_server)[0]

        assert isinstance(result, pa.Table)
        assert result.column_names == ["EngineRunnerTest1"]
        assert result.to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}

    def test_runner_single_feature(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        features = self.get_features(["EngineRunnerTest1"])
        result = self.basic_runner(features, modes, flight_server)[0]

        assert isinstance(result, pa.Table)
        assert result.column_names == ["EngineRunnerTest1"]
        assert result.to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}

    def test_runner_dependent_a_feature_only_child_given(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = self.get_features(["EngineRunnerTest2"])
        result = self.basic_runner(features, modes, flight_server)[0]

        assert result.to_pydict() == {"EngineRunnerTest2": [2, 4, 6]}

    def test_runner_dependent_single_feature_config_given(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = self.get_features(["EngineRunnerTest1"], {"config": "test"})
        result = self.basic_runner(features, modes, flight_server)[0]
        assert result.to_pydict() == {"EngineRunnerTest1": [2, 4, 6]}

    def test_runner_dependent_feature_config_given_1(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = self.get_features(["EngineRunnerTest2"], {"config": "test"})
        result = self.basic_runner(features, modes, flight_server)[0]
        assert result.to_pydict() == {"EngineRunnerTest2-test": [4, 8, 12]}

    def test_runner_dependent_feature_config_given_2(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = self.get_features(["EngineRunnerTest2", "EngineRunnerTest1"], {"config": "test"})
        result = self.basic_runner(features, modes, flight_server)
        assert result[0].to_pydict() == {"EngineRunnerTest1": [2, 4, 6]}
        assert result[1].to_pydict() == {"EngineRunnerTest2-test": [4, 8, 12]}

    def test_runner_dependent_feature_config_given_3(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = Features(
            [
                Feature(name="EngineRunnerTest1", options={"config": "test"}, initial_requested_data=True),
                Feature(name="EngineRunnerTest1", options={}, initial_requested_data=True),
            ]
        )

        result = self.basic_runner(features, modes, flight_server)
        assert (
            result[0].to_pydict() == {"EngineRunnerTest1": [2, 4, 6]}
            and result[1].to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}
        ) or (
            result[1].to_pydict() == {"EngineRunnerTest1": [2, 4, 6]}
            and result[0].to_pydict() == {"EngineRunnerTest1": [1, 2, 3]}
        ), "Inverted order because calculation order is not guaranteed."

    def test_runner_dependent_feature_config_given_4(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = Features(
            [
                Feature(name="EngineRunnerTest2", options={"config": "test"}, initial_requested_data=True),
                Feature(name="EngineRunnerTest2", options={}, initial_requested_data=True),
            ]
        )

        result = self.basic_runner(features, modes, flight_server)
        res1 = result[0].to_pydict()
        res2 = result[1].to_pydict()

        # order is not guaranteed
        no_config_res = res1 if res1.get("EngineRunnerTest2") else res2
        config_res = res2 if res2.get("EngineRunnerTest2-test") else res1
        assert no_config_res["EngineRunnerTest2"] != config_res["EngineRunnerTest2-test"]

        assert no_config_res["EngineRunnerTest2"] == [2, 4, 6]
        assert config_res["EngineRunnerTest2-test"] == [4, 8, 12]

    def test_runner_dependent_feature_config_given_5(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = Features(
            [Feature(name="EngineRunnerTest3", options={"config": "test"}, initial_requested_data=True)]
        )

        result = self.basic_runner(features, modes, flight_server)
        res = result[0].to_pydict()
        assert res == {"EngineRunnerTest3": [12, 24, 36]}

    def test_runner_features_4_multiple_colums_from_one_source(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = self.get_features(
            ["EngineRunnerTest4_1", "EngineRunnerTest4_2", "EngineRunnerTest4_3", "EngineRunnerTest4_0"]
        )
        result = self.basic_runner(features, modes, flight_server)[0]
        assert isinstance(result, pa.Table)

        assert result.to_pydict() == {
            "EngineRunnerTest4_0": [1.1, 2.2, 3.3],
            "EngineRunnerTest4_2": [4.4, 5.5, 6.6],
            "EngineRunnerTest4_3": [7.7, 8.8, 9.9],
            "EngineRunnerTest4_1": [10.1, 11.2, 12.3],
        }

    def test_runner_aggregated_feature_via_config_multiple(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        features = Features(
            [
                Feature(
                    name="sum_of_",
                    options={"sum": ("EngineRunnerTest4_0", "EngineRunnerTest4_1")},
                    initial_requested_data=True,
                )
            ]
        )
        result = self.basic_runner(features, modes, flight_server)[0]
        assert isinstance(result, pa.Table)
        assert result.to_pydict() == {
            "SumFeature_EngineRunnerTest4_0EngineRunnerTest4_1": [
                40.199999999999996,
                40.199999999999996,
                40.199999999999996,
            ]
        }
