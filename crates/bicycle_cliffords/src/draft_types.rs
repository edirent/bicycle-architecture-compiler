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

use std::{collections::BTreeMap, fmt, ops::BitXor};

use bicycle_common::{AutomorphismData, Pauli, TwoBases};
use serde::{Deserialize, Serialize};

pub const TAIL_QUBITS: u32 = 11;
pub const TAIL_MASK: u32 = (1 << TAIL_QUBITS) - 1;
pub const TAIL_SPACE_SIZE: u32 = 1 << (2 * TAIL_QUBITS);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Head {
    I,
    X,
    Y,
    Z,
}

impl Head {
    pub const NON_I: [Head; 3] = [Head::X, Head::Y, Head::Z];

    pub fn from_pauli(pauli: Pauli) -> Self {
        match pauli {
            Pauli::I => Self::I,
            Pauli::X => Self::X,
            Pauli::Y => Self::Y,
            Pauli::Z => Self::Z,
        }
    }

    pub fn as_pauli(self) -> Pauli {
        match self {
            Self::I => Pauli::I,
            Self::X => Pauli::X,
            Self::Y => Pauli::Y,
            Self::Z => Pauli::Z,
        }
    }

    pub fn to_char(self) -> char {
        match self {
            Self::I => 'I',
            Self::X => 'X',
            Self::Y => 'Y',
            Self::Z => 'Z',
        }
    }

    pub fn from_char(value: char) -> Result<Self, String> {
        match value.to_ascii_uppercase() {
            'I' => Ok(Self::I),
            'X' => Ok(Self::X),
            'Y' => Ok(Self::Y),
            'Z' => Ok(Self::Z),
            c => Err(format!("Invalid head symbol: {c}")),
        }
    }

    pub fn xor(self, rhs: Self) -> Self {
        let (self_x, self_z) = self.symplectic_bits();
        let (rhs_x, rhs_z) = rhs.symplectic_bits();
        Self::from_symplectic_bits(self_x ^ rhs_x, self_z ^ rhs_z)
    }

    fn symplectic_bits(self) -> (u8, u8) {
        match self {
            Self::I => (0, 0),
            Self::X => (1, 0),
            Self::Y => (1, 1),
            Self::Z => (0, 1),
        }
    }

    fn from_symplectic_bits(x: u8, z: u8) -> Self {
        match (x, z) {
            (0, 0) => Self::I,
            (1, 0) => Self::X,
            (1, 1) => Self::Y,
            (0, 1) => Self::Z,
            _ => unreachable!("Head symplectic bits must be in {{0,1}}"),
        }
    }
}

impl fmt::Display for Head {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Tail11(pub u32);

impl Tail11 {
    pub fn new(bits: u32) -> Self {
        assert!(
            bits < TAIL_SPACE_SIZE,
            "Tail bits exceed 11-qubit symplectic range"
        );
        Self(bits)
    }

    pub fn x_bits(self) -> u32 {
        self.0 & TAIL_MASK
    }

    pub fn z_bits(self) -> u32 {
        self.0 >> TAIL_QUBITS
    }

    pub fn from_paulis(paulis: [Pauli; TAIL_QUBITS as usize]) -> Self {
        let mut x_bits = 0u32;
        let mut z_bits = 0u32;
        for (i, pauli) in paulis.into_iter().enumerate() {
            let shift = i as u32;
            match pauli {
                Pauli::I => {}
                Pauli::X => x_bits |= 1 << shift,
                Pauli::Z => z_bits |= 1 << shift,
                Pauli::Y => {
                    x_bits |= 1 << shift;
                    z_bits |= 1 << shift;
                }
            }
        }
        Self::new((z_bits << TAIL_QUBITS) | x_bits)
    }

    pub fn to_paulis(self) -> [Pauli; TAIL_QUBITS as usize] {
        let mut out = [Pauli::I; TAIL_QUBITS as usize];
        let x_bits = self.x_bits();
        let z_bits = self.z_bits();
        for (i, pauli) in out.iter_mut().enumerate() {
            let bit = 1 << i;
            let has_x = x_bits & bit != 0;
            let has_z = z_bits & bit != 0;
            *pauli = match (has_x, has_z) {
                (false, false) => Pauli::I,
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
            };
        }
        out
    }

    pub fn from_label(label: &str) -> Result<Self, String> {
        if label.chars().count() != TAIL_QUBITS as usize {
            return Err(format!(
                "Expected {} Pauli symbols in tail label, got {}",
                TAIL_QUBITS,
                label.chars().count()
            ));
        }
        let mut out = [Pauli::I; TAIL_QUBITS as usize];
        for (i, c) in label.chars().enumerate() {
            out[i] = Pauli::try_from(&c)?;
        }
        Ok(Self::from_paulis(out))
    }

    pub fn to_label(self) -> String {
        self.to_paulis().iter().map(|p| format!("{p}")).collect()
    }

    pub fn commute_symplectic_inner(self, rhs: Self) -> u32 {
        ((self.x_bits() & rhs.z_bits()).count_ones() + (self.z_bits() & rhs.x_bits()).count_ones())
            % 2
    }

    pub fn commutes_with(self, rhs: Self) -> bool {
        self.commute_symplectic_inner(rhs) == 0
    }

    pub fn anticommutes_with(self, rhs: Self) -> bool {
        !self.commutes_with(rhs)
    }

    pub fn xor(self, rhs: Self) -> Self {
        Self::new(self.0 ^ rhs.0)
    }
}

impl fmt::Display for Tail11 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_label())
    }
}

impl BitXor for Tail11 {
    type Output = Tail11;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRowMetadata {
    pub measured_pauli_bits: u32,
    pub logical: TwoBases,
    pub automorphism: AutomorphismData,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRow {
    pub index: usize,
    pub head: Head,
    pub tail: Tail11,
    pub metadata: Option<NativeRowMetadata>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum RuleClass {
    P0I,
    P0Q,
    P2,
    P3Delta4,
    P3Delta6,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct StateKey {
    pub tail: Tail11,
    pub head: Head,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct TransitionRow {
    pub prev_head: Head,
    pub q_basis: Head,
    pub next_head: Head,
    pub delta_stage_cost: u8,
    pub delta_local_measurement_count: u8,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransitionOutcome {
    pub next_head: Head,
    pub delta_stage_cost: u32,
    pub delta_local_measurement_count: u8,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct P0IWitness {
    pub native_index: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct P0QWitness {
    pub q_basis: Head,
    pub native_index: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct P2Witness {
    pub q_basis: Head,
    pub left_tail: Tail11,
    pub right_tail: Tail11,
    pub left_native_index: usize,
    pub right_native_index: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct P3Witness {
    pub prev_head: Head,
    pub prev_tail: Tail11,
    pub q_basis: Head,
    pub axis_tail: Tail11,
    pub next_head: Head,
    pub delta_stage_cost: u32,
    pub delta_local_measurement_count: u8,
    pub prev_state: StateKey,
    pub native_index: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum InsertionWitness {
    P0I(P0IWitness),
    P0Q(P0QWitness),
    P2(P2Witness),
    P3(P3Witness),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateEntry {
    pub key: StateKey,
    pub insertion_stage_cost: u32,
    pub local_measurement_count: u8,
    pub witness: InsertionWitness,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct LibraryEntry {
    pub tail: Tail11,
    pub resulting_head: Head,
    pub first_rule: RuleClass,
    pub insertion_stage_cost: u32,
    pub local_measurement_count: u8,
    pub witness: InsertionWitness,
}

#[derive(Debug, Clone, Eq, PartialEq, Default, Serialize, Deserialize)]
pub struct DraftHistograms {
    pub direct_use_cost_hist: BTreeMap<u32, u64>,
    pub insertion_stage_cost_hist: BTreeMap<u32, u64>,
    pub local_measurement_count_hist: BTreeMap<u8, u64>,
    pub rule_hist: BTreeMap<RuleClass, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftBuildResult {
    pub native_rows: Vec<NativeRow>,
    pub states: std::collections::HashMap<StateKey, StateEntry>,
    pub library: std::collections::HashMap<Tail11, LibraryEntry>,
    pub histograms: DraftHistograms,
    pub full_coverage: bool,
}

impl DraftBuildResult {
    pub fn library_size(&self) -> usize {
        self.library.len()
    }
}
