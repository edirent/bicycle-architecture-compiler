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

use std::fmt::Display;

use bicycle_common::Pauli;
use fixed::types::I32F96;

use bicycle_cliffords::CompleteMeasurementTable;
use serde::{Deserialize, Serialize};

use crate::{
    architecture::PathArchitecture,
    compile::{self, CompiledMeasurementPlan},
    operation::Operation,
};

pub type AnglePrecision = I32F96;

/// A PBC program operation
/// Consider replacing the angle with a rational to improve precision.
/// But f64 has 52-bit mantissa, so seems sufficient for all practical purposes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PbcOperation {
    Measurement {
        basis: Vec<Pauli>,
        flip_result: bool,
    },
    Rotation {
        basis: Vec<Pauli>,
        angle: AnglePrecision,
    },
}

impl PbcOperation {
    pub fn rotation(basis: Vec<Pauli>, angle: f64) -> Self {
        Self::Rotation {
            basis,
            angle: AnglePrecision::from_num(angle),
        }
    }
    pub fn compile(
        &self,
        architecture: &PathArchitecture,
        measurement_table: &CompleteMeasurementTable,
        accuracy: AnglePrecision,
    ) -> Vec<Operation> {
        match self {
            PbcOperation::Measurement { basis, flip_result } => {
                assert!(
                    !*flip_result,
                    "flip_result=true is not supported by legacy PbcOperation::compile; use PbcOperation::compile_measurement_plan"
                );
                compile::compile_measurement(architecture, measurement_table, basis.to_vec())
            }
            PbcOperation::Rotation { basis, angle } => compile::compile_rotation(
                architecture,
                measurement_table,
                basis.to_vec(),
                *angle,
                accuracy,
            ),
        }
    }

    pub fn compile_measurement_plan(
        &self,
        architecture: &PathArchitecture,
        measurement_table: &CompleteMeasurementTable,
    ) -> Option<CompiledMeasurementPlan> {
        match self {
            PbcOperation::Measurement { basis, flip_result } => {
                Some(compile::compile_measurement_plan(
                    architecture,
                    measurement_table,
                    basis.to_vec(),
                    *flip_result,
                ))
            }
            PbcOperation::Rotation { .. } => None,
        }
    }

    pub fn basis(&self) -> &Vec<Pauli> {
        match self {
            PbcOperation::Measurement {
                basis,
                flip_result: _,
            } => basis,
            PbcOperation::Rotation { basis, angle: _ } => basis,
        }
    }
}

impl Display for PbcOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PbcOperation::Measurement { basis, flip_result } => {
                write!(
                    f,
                    "Measurement([{}],",
                    basis
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )?;
                if *flip_result {
                    write!(f, "flipped)")
                } else {
                    write!(f, "regular)")
                }
            }
            PbcOperation::Rotation { basis, angle } => {
                write!(
                    f,
                    "Rotation([{}],{})",
                    basis
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                    angle
                )
            }
        }
    }
}
