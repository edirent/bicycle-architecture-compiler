// Copyright contributors to the Bicycle Architecture Compiler project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod architecture;
mod basis_changer;
mod compile;
pub mod language;
pub mod operation;
pub mod optimize;
pub mod small_angle;

use std::{error::Error, path::Path};

pub use architecture::PathArchitecture;
use bicycle_cliffords::CompleteMeasurementTable;
pub use compile::{CompiledMeasurementPlan, X1DotX8CheckReport, check_x1_dot_x8_compilation};

pub fn deserialize_table(cache_path: &Path) -> Result<CompleteMeasurementTable, Box<dyn Error>> {
    let read = std::fs::read(cache_path)?;
    Ok(bitcode::deserialize::<CompleteMeasurementTable>(&read)?)
}

#[cfg(test)]
mod test {

    use std::error::Error;

    use crate::language::{AnglePrecision, PbcOperation};

    use super::*;
    use bicycle_cliffords::{
        MeasurementTableBuilder, TWOGROSS_MEASUREMENT, native_measurement::NativeMeasurement,
    };
    use bicycle_common::Pauli;

    fn build_table() -> Result<CompleteMeasurementTable, String> {
        let mut builder =
            MeasurementTableBuilder::new(NativeMeasurement::all(), TWOGROSS_MEASUREMENT);
        builder.build();
        builder.complete()
    }

    #[test]
    fn integration_test_rotation_rejects_nontrivial_clifford_corrections()
    -> Result<(), Box<dyn Error>> {
        let program = r#"[
                                    {
                                        "Rotation": {
                                        "basis": [
                                            "X",
                                            "X",
                                            "I",
                                            "I",
                                            "I",
                                            "I",
                                            "I",
                                            "I",
                                            "I",
                                            "I",
                                            "I",
                                            "Y"
                                        ],
                                        "angle": "0.125"
                                        }
                                    }
                                ]"#;
        let parsed: Vec<PbcOperation> = serde_json::from_str(program)?;
        dbg!(&parsed);
        assert_eq!(1, parsed.len());

        let measurement_table = build_table()?;

        let architecture = PathArchitecture { data_blocks: 2 };
        let panic = std::panic::catch_unwind(|| {
            let _compiled: Vec<_> = parsed
                .into_iter()
                .flat_map(|op| {
                    op.compile(
                        &architecture,
                        &measurement_table,
                        AnglePrecision::lit("1e-16"),
                    )
                })
                .collect();
        })
        .expect_err("rotation with nontrivial clifford corrections should be rejected");

        let message = if let Some(msg) = panic.downcast_ref::<String>() {
            msg.clone()
        } else if let Some(msg) = panic.downcast_ref::<&'static str>() {
            (*msg).to_string()
        } else {
            "<non-string panic payload>".to_string()
        };
        assert!(message.contains("nontrivial Clifford corrections"));

        Ok(())
    }

    #[test]
    fn integration_test_measurement_plan_preserves_flip_sidecar() -> Result<(), Box<dyn Error>> {
        let measurement_table = build_table()?;
        let architecture = PathArchitecture { data_blocks: 1 };

        let with_flip = PbcOperation::Measurement {
            basis: vec![Pauli::X],
            flip_result: true,
        };
        let plan = with_flip
            .compile_measurement_plan(&architecture, &measurement_table)
            .expect("measurement op should produce a sidecar plan");
        assert!(plan.logical_result_flip);

        let without_flip = PbcOperation::Measurement {
            basis: vec![Pauli::X],
            flip_result: false,
        };
        assert_eq!(
            without_flip.compile(
                &architecture,
                &measurement_table,
                AnglePrecision::lit("1e-16")
            ),
            plan.ops
        );

        Ok(())
    }

    #[test]
    #[should_panic(
        expected = "flip_result=true is not supported by legacy PbcOperation::compile; use PbcOperation::compile_measurement_plan"
    )]
    fn integration_test_legacy_compile_rejects_flip_result() {
        let measurement_table = build_table().expect("table should build");
        let architecture = PathArchitecture { data_blocks: 1 };
        let op = PbcOperation::Measurement {
            basis: vec![Pauli::X],
            flip_result: true,
        };

        let _ = op.compile(
            &architecture,
            &measurement_table,
            AnglePrecision::lit("1e-16"),
        );
    }
}
