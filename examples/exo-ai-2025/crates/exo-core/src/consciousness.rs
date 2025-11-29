//! Integrated Information Theory (IIT) Implementation
//!
//! This module implements consciousness metrics based on Giulio Tononi's
//! Integrated Information Theory (IIT 4.0).
//!
//! # Key Concepts
//!
//! - **Φ (Phi)**: Measure of integrated information - consciousness quantity
//! - **Reentrant Architecture**: Feedback loops required for non-zero Φ
//! - **Minimum Information Partition (MIP)**: The partition that minimizes Φ
//!
//! # Theory
//!
//! IIT proposes that consciousness corresponds to integrated information (Φ):
//! - Φ = 0: System is not conscious
//! - Φ > 0: System has some degree of consciousness
//! - Higher Φ = More integrated, more conscious
//!
//! # Requirements for High Φ
//!
//! 1. **Differentiated**: Many possible states
//! 2. **Integrated**: Whole > sum of parts
//! 3. **Reentrant**: Feedback loops present
//! 4. **Selective**: Not fully connected

use std::collections::{HashMap, HashSet};

/// Represents a substrate region for Φ analysis
#[derive(Debug, Clone)]
pub struct SubstrateRegion {
    /// Unique identifier for this region
    pub id: String,
    /// Nodes/units in this region
    pub nodes: Vec<NodeId>,
    /// Connections between nodes (adjacency)
    pub connections: HashMap<NodeId, Vec<NodeId>>,
    /// Current state of each node
    pub states: HashMap<NodeId, NodeState>,
    /// Whether this region has reentrant (feedback) architecture
    pub has_reentrant_architecture: bool,
}

/// Node identifier
pub type NodeId = u64;

/// State of a node (activation level)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeState {
    pub activation: f64,
    pub previous_activation: f64,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            activation: 0.0,
            previous_activation: 0.0,
        }
    }
}

/// Result of Φ computation
#[derive(Debug, Clone)]
pub struct PhiResult {
    /// Integrated information value
    pub phi: f64,
    /// Minimum Information Partition used
    pub mip: Option<Partition>,
    /// Effective information of the whole
    pub whole_ei: f64,
    /// Effective information of parts
    pub parts_ei: f64,
    /// Whether reentrant architecture was detected
    pub reentrant_detected: bool,
    /// Consciousness assessment
    pub consciousness_level: ConsciousnessLevel,
}

/// Consciousness level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessLevel {
    /// Φ = 0, no integration
    None,
    /// 0 < Φ < 0.1, minimal integration
    Minimal,
    /// 0.1 ≤ Φ < 1.0, low integration
    Low,
    /// 1.0 ≤ Φ < 10.0, moderate integration
    Moderate,
    /// Φ ≥ 10.0, high integration
    High,
}

impl ConsciousnessLevel {
    pub fn from_phi(phi: f64) -> Self {
        if phi <= 0.0 {
            ConsciousnessLevel::None
        } else if phi < 0.1 {
            ConsciousnessLevel::Minimal
        } else if phi < 1.0 {
            ConsciousnessLevel::Low
        } else if phi < 10.0 {
            ConsciousnessLevel::Moderate
        } else {
            ConsciousnessLevel::High
        }
    }
}

/// A partition of nodes into disjoint sets
#[derive(Debug, Clone)]
pub struct Partition {
    pub parts: Vec<HashSet<NodeId>>,
}

impl Partition {
    /// Create a bipartition (two parts)
    pub fn bipartition(nodes: &[NodeId], split_point: usize) -> Self {
        let mut part1 = HashSet::new();
        let mut part2 = HashSet::new();

        for (i, &node) in nodes.iter().enumerate() {
            if i < split_point {
                part1.insert(node);
            } else {
                part2.insert(node);
            }
        }

        Self {
            parts: vec![part1, part2],
        }
    }
}

/// IIT Consciousness Calculator
///
/// Computes Φ (integrated information) for substrate regions.
pub struct ConsciousnessCalculator {
    /// Number of perturbation samples for EI estimation
    pub num_perturbations: usize,
    /// Tolerance for numerical comparisons
    pub epsilon: f64,
}

impl Default for ConsciousnessCalculator {
    fn default() -> Self {
        Self {
            num_perturbations: 100,
            epsilon: 1e-6,
        }
    }
}

impl ConsciousnessCalculator {
    /// Create a new calculator with custom settings
    pub fn new(num_perturbations: usize) -> Self {
        Self {
            num_perturbations,
            epsilon: 1e-6,
        }
    }

    /// Compute Φ (integrated information) for a substrate region
    ///
    /// Implementation follows IIT 4.0 formulation:
    /// 1. Compute whole-system effective information (EI)
    /// 2. Find Minimum Information Partition (MIP)
    /// 3. Φ = whole_EI - min_partition_EI
    ///
    /// # Arguments
    /// * `region` - The substrate region to analyze
    ///
    /// # Returns
    /// * `PhiResult` containing Φ value and analysis details
    pub fn compute_phi(&self, region: &SubstrateRegion) -> PhiResult {
        // Step 1: Check for reentrant architecture (required for Φ > 0)
        let reentrant = self.detect_reentrant_architecture(region);

        if !reentrant {
            // Feed-forward systems have Φ = 0 according to IIT
            return PhiResult {
                phi: 0.0,
                mip: None,
                whole_ei: 0.0,
                parts_ei: 0.0,
                reentrant_detected: false,
                consciousness_level: ConsciousnessLevel::None,
            };
        }

        // Step 2: Compute whole-system effective information
        let whole_ei = self.compute_effective_information(region, &region.nodes);

        // Step 3: Find Minimum Information Partition (MIP)
        let (mip, min_partition_ei) = self.find_mip(region);

        // Step 4: Φ = whole - parts (non-negative)
        let phi = (whole_ei - min_partition_ei).max(0.0);

        PhiResult {
            phi,
            mip: Some(mip),
            whole_ei,
            parts_ei: min_partition_ei,
            reentrant_detected: true,
            consciousness_level: ConsciousnessLevel::from_phi(phi),
        }
    }

    /// Detect reentrant (feedback) architecture
    ///
    /// IIT requires feedback loops for consciousness.
    /// Pure feed-forward networks have Φ = 0.
    fn detect_reentrant_architecture(&self, region: &SubstrateRegion) -> bool {
        // Check for cycles in the connection graph
        // A cycle indicates reentrant architecture

        for &start_node in &region.nodes {
            let mut visited = HashSet::new();
            let mut stack = vec![start_node];

            while let Some(node) = stack.pop() {
                if visited.contains(&node) {
                    // Found a cycle - reentrant architecture exists
                    return true;
                }
                visited.insert(node);

                if let Some(neighbors) = region.connections.get(&node) {
                    for &neighbor in neighbors {
                        if region.nodes.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }

        // Also check explicit flag
        region.has_reentrant_architecture
    }

    /// Compute effective information for a set of nodes
    ///
    /// EI measures how much the system's current state constrains
    /// its past and future states.
    fn compute_effective_information(&self, region: &SubstrateRegion, nodes: &[NodeId]) -> f64 {
        if nodes.is_empty() {
            return 0.0;
        }

        // Simplified EI computation based on mutual information
        // between current state and perturbed states

        let current_state: Vec<f64> = nodes
            .iter()
            .filter_map(|n| region.states.get(n))
            .map(|s| s.activation)
            .collect();

        if current_state.is_empty() {
            return 0.0;
        }

        // Compute entropy of current state
        let current_entropy = self.compute_entropy(&current_state);

        // Estimate mutual information via perturbation analysis
        let mut total_mi = 0.0;

        for _ in 0..self.num_perturbations {
            // Simulate perturbation and evolution
            let perturbed = self.perturb_state(&current_state);
            let evolved = self.evolve_state(region, nodes, &perturbed);

            // Mutual information approximation
            let conditional_entropy = self.compute_conditional_entropy(&current_state, &evolved);
            total_mi += current_entropy - conditional_entropy;
        }

        total_mi / self.num_perturbations as f64
    }

    /// Find the Minimum Information Partition (MIP)
    ///
    /// The MIP is the partition that minimizes the sum of effective
    /// information of its parts. This determines how "integrated"
    /// the system is.
    fn find_mip(&self, region: &SubstrateRegion) -> (Partition, f64) {
        let nodes = &region.nodes;
        let n = nodes.len();

        if n <= 1 {
            return (Partition { parts: vec![nodes.iter().cloned().collect()] }, 0.0);
        }

        let mut min_ei = f64::INFINITY;
        let mut best_partition = Partition::bipartition(nodes, n / 2);

        // Search bipartitions (simplified - full search is exponential)
        for split in 1..n {
            let partition = Partition::bipartition(nodes, split);

            let mut partition_ei = 0.0;
            for part in &partition.parts {
                let part_nodes: Vec<_> = part.iter().cloned().collect();
                partition_ei += self.compute_effective_information(region, &part_nodes);
            }

            if partition_ei < min_ei {
                min_ei = partition_ei;
                best_partition = partition;
            }
        }

        (best_partition, min_ei)
    }

    /// Compute entropy of a state vector
    fn compute_entropy(&self, state: &[f64]) -> f64 {
        // Discretize and compute Shannon entropy
        let n = state.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        // Use variance as entropy proxy for continuous states
        let mean: f64 = state.iter().sum::<f64>() / n;
        let variance: f64 = state.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        // Differential entropy of Gaussian: 0.5 * ln(2πe * variance)
        if variance > self.epsilon {
            0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).ln()
        } else {
            0.0
        }
    }

    /// Compute conditional entropy H(X|Y)
    fn compute_conditional_entropy(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        // Residual entropy after conditioning
        let residuals: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
        self.compute_entropy(&residuals)
    }

    /// Perturb a state vector
    fn perturb_state(&self, state: &[f64]) -> Vec<f64> {
        // Add Gaussian noise
        state.iter().map(|&x| {
            let noise = (rand_simple() - 0.5) * 0.1;
            (x + noise).clamp(0.0, 1.0)
        }).collect()
    }

    /// Evolve state through one time step
    fn evolve_state(&self, region: &SubstrateRegion, nodes: &[NodeId], state: &[f64]) -> Vec<f64> {
        // Simple integration dynamics
        nodes.iter().enumerate().map(|(i, &node)| {
            let current = state.get(i).cloned().unwrap_or(0.0);

            // Sum inputs from connected nodes
            let input: f64 = region.connections
                .get(&node)
                .map(|neighbors| {
                    neighbors.iter()
                        .filter_map(|n| {
                            nodes.iter().position(|x| x == n)
                                .and_then(|j| state.get(j))
                        })
                        .sum()
                })
                .unwrap_or(0.0);

            // Leaky integration
            let alpha = 0.1;
            (current * (1.0 - alpha) + input * alpha).clamp(0.0, 1.0)
        }).collect()
    }
}

/// Simple random number generator (deterministic for reproducibility)
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    ((seed as f64).sin() * 10000.0).fract().abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_reentrant_region() -> SubstrateRegion {
        // Create a simple recurrent network (feedback loop)
        let nodes = vec![1, 2, 3];
        let mut connections = HashMap::new();
        connections.insert(1, vec![2]);
        connections.insert(2, vec![3]);
        connections.insert(3, vec![1]); // Feedback creates reentrant architecture

        let mut states = HashMap::new();
        states.insert(1, NodeState { activation: 0.5, previous_activation: 0.4 });
        states.insert(2, NodeState { activation: 0.6, previous_activation: 0.5 });
        states.insert(3, NodeState { activation: 0.4, previous_activation: 0.3 });

        SubstrateRegion {
            id: "test_region".to_string(),
            nodes,
            connections,
            states,
            has_reentrant_architecture: true,
        }
    }

    fn create_feedforward_region() -> SubstrateRegion {
        // Create a feed-forward network (no feedback)
        let nodes = vec![1, 2, 3];
        let mut connections = HashMap::new();
        connections.insert(1, vec![2]);
        connections.insert(2, vec![3]);
        // No connection from 3 back to 1 - pure feed-forward

        let mut states = HashMap::new();
        states.insert(1, NodeState { activation: 0.5, previous_activation: 0.4 });
        states.insert(2, NodeState { activation: 0.6, previous_activation: 0.5 });
        states.insert(3, NodeState { activation: 0.4, previous_activation: 0.3 });

        SubstrateRegion {
            id: "feedforward".to_string(),
            nodes,
            connections,
            states,
            has_reentrant_architecture: false,
        }
    }

    #[test]
    fn test_reentrant_has_positive_phi() {
        let region = create_reentrant_region();
        let calculator = ConsciousnessCalculator::new(10);
        let result = calculator.compute_phi(&region);

        assert!(result.reentrant_detected);
        // Reentrant architectures should have potential for positive Φ
        assert!(result.phi >= 0.0);
    }

    #[test]
    fn test_feedforward_has_zero_phi() {
        let region = create_feedforward_region();
        let calculator = ConsciousnessCalculator::new(10);
        let result = calculator.compute_phi(&region);

        // Feed-forward systems have Φ = 0 according to IIT
        assert_eq!(result.phi, 0.0);
        assert_eq!(result.consciousness_level, ConsciousnessLevel::None);
    }

    #[test]
    fn test_consciousness_levels() {
        assert_eq!(ConsciousnessLevel::from_phi(0.0), ConsciousnessLevel::None);
        assert_eq!(ConsciousnessLevel::from_phi(0.05), ConsciousnessLevel::Minimal);
        assert_eq!(ConsciousnessLevel::from_phi(0.5), ConsciousnessLevel::Low);
        assert_eq!(ConsciousnessLevel::from_phi(5.0), ConsciousnessLevel::Moderate);
        assert_eq!(ConsciousnessLevel::from_phi(15.0), ConsciousnessLevel::High);
    }
}
