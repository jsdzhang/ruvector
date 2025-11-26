# RuVector

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/npm/v/ruvector.svg)](https://www.npmjs.com/package/ruvector)
[![npm downloads](https://img.shields.io/npm/dm/ruvector.svg)](https://www.npmjs.com/package/ruvector)

**A distributed vector database that learns.** Store embeddings, query with Cypher, scale horizontally, and let the index improve itself through Graph Neural Networks.

```bash
npx ruvector
```

> **All-in-One Package**: The `ruvector` package includes everything — vector search, graph queries, GNN layers, distributed clustering, AI routing, and WASM support. No additional packages needed.

## What Problem Does RuVector Solve?

Traditional vector databases just store and search. When you ask "find similar items," they return results but never get smarter.

**RuVector is different:**

1. **Store vectors** like any vector DB (embeddings from OpenAI, Cohere, etc.)
2. **Query with Cypher** like Neo4j (`MATCH (a)-[:SIMILAR]->(b) RETURN b`)
3. **The index learns** — GNN layers make search results improve over time
4. **Route AI requests** — Semantic routing and FastGRNN neural inference for LLM optimization
5. **Compress automatically** — 2-32x memory reduction with adaptive tiered compression
6. **Run anywhere** — Node.js, browser (WASM), or native Rust

## Quick Start

### Installation

```bash
# Install the package
npm install ruvector

# Or try instantly without installing
npx ruvector
```

### Basic Usage

```javascript
const ruvector = require('ruvector');

// Create a vector database
const db = new ruvector.VectorDB(384); // 384 dimensions

// Insert vectors with metadata
db.insert('doc1', embedding1, { title: 'Introduction', category: 'tech' });
db.insert('doc2', embedding2, { title: 'Advanced Topics', category: 'tech' });

// Search for similar vectors
const results = db.search(queryEmbedding, 10);
console.log(results); // Top 10 similar documents

// Filter by metadata
const filtered = db.search(queryEmbedding, 10, { category: 'tech' });
```

### Graph Queries (Cypher)

```javascript
const { GraphDB } = require('ruvector');

const graph = new GraphDB();

// Create nodes and relationships
graph.execute("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})");
graph.execute("CREATE (b)-[:WORKS_AT]->(c:Company {name: 'TechCorp'})");

// Query relationships
const friends = graph.execute("MATCH (p:Person)-[:KNOWS]->(friend) RETURN friend.name");
const colleagues = graph.execute(`
  MATCH (p:Person {name: 'Alice'})-[:KNOWS]->(friend)-[:WORKS_AT]->(company)
  RETURN friend.name, company.name
`);
```

### GNN-Enhanced Search

```javascript
const { GNNLayer } = require('ruvector');

// Create a GNN layer (input_dim, output_dim, num_heads)
const layer = new GNNLayer(384, 512, 4);

// Enhance query with graph context
const query = getQueryEmbedding();
const neighbors = getNeighborEmbeddings();
const weights = computeEdgeWeights();

const enhanced = layer.forward(query, neighbors, weights);
// Use enhanced embedding for better search results
```

### Compression (2-32x Memory Savings)

```javascript
const { compress, decompress, CompressionTier } = require('ruvector');

// Automatic tier selection based on quality threshold
const compressed = compress(embedding, 0.3); // 30% quality threshold

// Or specify tier explicitly
const pq8 = compress(embedding, CompressionTier.PQ8);   // 8x compression
const pq4 = compress(embedding, CompressionTier.PQ4);   // 16x compression
const binary = compress(embedding, CompressionTier.Binary); // 32x compression

// Decompress when needed
const restored = decompress(compressed);
```

### AI Agent Routing (Tiny Dancer)

```javascript
const { Router } = require('ruvector');

// Create router for AI model selection
const router = new Router({
  confidenceThreshold: 0.85,
  maxUncertainty: 0.15
});

// Route to optimal model based on query complexity
const candidates = [
  { id: 'gpt-4', embedding: gpt4Embedding, cost: 0.03 },
  { id: 'gpt-3.5', embedding: gpt35Embedding, cost: 0.002 },
  { id: 'claude', embedding: claudeEmbedding, cost: 0.015 }
];

const decision = router.route(queryEmbedding, candidates);
console.log(decision);
// { candidateId: 'gpt-3.5', confidence: 0.92, useLightweight: true }
```

## CLI Usage

```bash
# Show system info and backend status
npx ruvector info

# Initialize a new index
npx ruvector init my-index.bin --dimension 384 --type hnsw

# Insert vectors from JSON file
npx ruvector insert my-index.bin vectors.json

# Search with a query vector
npx ruvector search my-index.bin --query "[0.1, 0.2, ...]" -k 10

# Show index statistics
npx ruvector stats my-index.bin

# Run performance benchmarks
npx ruvector benchmark --dimension 384 --num-vectors 10000
```

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Vector Search** | HNSW index, <0.5ms latency, SIMD acceleration |
| **Cypher Queries** | `MATCH`, `WHERE`, `CREATE`, `RETURN` — Neo4j syntax |
| **GNN Layers** | Multi-head attention on graph topology |
| **Hyperedges** | Connect 3+ nodes simultaneously |
| **Metadata Filtering** | Combine semantic + structured search |
| **Collections** | Namespace isolation, multi-tenancy |

### AI & ML

| Feature | Description |
|---------|-------------|
| **Tensor Compression** | f32→f16→PQ8→PQ4→Binary (2-32x reduction) |
| **Differentiable Search** | Soft attention k-NN for end-to-end training |
| **Semantic Router** | Route queries to optimal endpoints |
| **Tiny Dancer** | FastGRNN neural inference for LLM cost optimization |

### Platform Support

| Platform | Package | Notes |
|----------|---------|-------|
| **Node.js** | `ruvector` | Native bindings via napi-rs |
| **Browser** | `@ruvector/wasm` | Full WASM support |
| **Bun** | `ruvector` | Native bindings |
| **Deno** | `@ruvector/wasm` | WASM fallback |

## Benchmarks

| Operation | Dimensions | Time | Throughput |
|-----------|------------|------|------------|
| **HNSW Search (k=10)** | 384 | 61µs | 16,400 QPS |
| **HNSW Search (k=100)** | 384 | 164µs | 6,100 QPS |
| **Cosine Distance** | 1536 | 143ns | 7M ops/sec |
| **Dot Product** | 384 | 33ns | 30M ops/sec |
| **Insert** | 384 | 20µs | 50,000/sec |

Run your own benchmarks:
```bash
npx ruvector benchmark --dimension 384 --num-vectors 10000
```

## npm Packages

| Package | Description |
|---------|-------------|
| [`ruvector`](https://www.npmjs.com/package/ruvector) | All-in-one package (recommended) |
| [`@ruvector/wasm`](https://www.npmjs.com/package/@ruvector/wasm) | Browser/WASM bindings |
| [`@ruvector/graph`](https://www.npmjs.com/package/@ruvector/graph) | Graph database with Cypher |
| [`@ruvector/gnn`](https://www.npmjs.com/package/@ruvector/gnn) | Graph Neural Network layers |
| [`@ruvector/tiny-dancer`](https://www.npmjs.com/package/@ruvector/tiny-dancer) | AI agent routing (FastGRNN) |
| [`@ruvector/router`](https://www.npmjs.com/package/@ruvector/router) | Semantic routing engine |

```bash
# Install all-in-one (recommended)
npm install ruvector

# Or install specific packages
npm install @ruvector/graph @ruvector/gnn
```

## API Reference

### VectorDB

```typescript
class VectorDB {
  constructor(dimension: number, options?: VectorDBOptions);

  insert(id: string, values: number[], metadata?: object): void;
  insertBatch(vectors: Vector[]): void;
  search(query: number[], k?: number, filter?: object): SearchResult[];
  get(id: string): Vector | null;
  delete(id: string): boolean;
  save(path: string): void;
  static load(path: string): VectorDB;
}
```

### GraphDB

```typescript
class GraphDB {
  constructor();

  execute(cypher: string): QueryResult;
  createNode(label: string, properties: object): string;
  createRelationship(from: string, to: string, type: string): void;
  createHyperedge(nodeIds: string[], type: string): string;
}
```

### GNNLayer

```typescript
class GNNLayer {
  constructor(inputDim: number, outputDim: number, numHeads: number);

  forward(query: number[], neighbors: number[][], weights: number[]): number[];
  train(data: TrainingData, epochs: number): void;
}
```

### Router (Tiny Dancer)

```typescript
class Router {
  constructor(config?: RouterConfig);

  route(query: number[], candidates: Candidate[]): RoutingDecision;
  reloadModel(): void;
  circuitBreakerStatus(): 'closed' | 'open' | 'half-open';
}
```

## Use Cases

### RAG (Retrieval-Augmented Generation)

```javascript
const ruvector = require('ruvector');

async function ragQuery(question) {
  const questionEmbedding = await embed(question);
  const context = db.search(questionEmbedding, 5);

  const prompt = `
    Context: ${context.map(c => c.metadata.text).join('\n')}

    Question: ${question}
    Answer:
  `;

  return await llm.complete(prompt);
}
```

### Recommendation System

```javascript
const { GraphDB } = require('ruvector');

const graph = new GraphDB();

// Find recommendations based on user behavior
const recommendations = graph.execute(`
  MATCH (user:User {id: $userId})-[:VIEWED]->(item:Product)
  MATCH (item)-[:SIMILAR_TO]->(rec:Product)
  WHERE NOT (user)-[:VIEWED]->(rec)
  RETURN rec ORDER BY rec.score DESC LIMIT 10
`);
```

### Semantic Search with Filters

```javascript
const results = db.search(queryEmbedding, 20, {
  category: 'electronics',
  price: { $lt: 500 },
  inStock: true
});
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       ruvector                          │
│              (All-in-One npm Package)                   │
├─────────────┬─────────────┬─────────────┬──────────────┤
│  VectorDB   │   GraphDB   │  GNNLayer   │   Router     │
│  (Search)   │  (Cypher)   │  (ML)       │ (AI Routing) │
└─────────────┴─────────────┴─────────────┴──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │  Native │       │  WASM   │       │  FFI    │
   │ (napi)  │       │(wasm32) │       │  (C)    │
   └─────────┘       └─────────┘       └─────────┘
```

The package automatically selects the best available backend (native > WASM).

## Comparison

| Feature | RuVector | Pinecone | Qdrant | ChromaDB |
|---------|----------|----------|--------|----------|
| **Latency** | **61µs** | ~2ms | ~1ms | ~50ms |
| **Graph Queries** | ✅ Cypher | ❌ | ❌ | ❌ |
| **Self-Learning** | ✅ GNN | ❌ | ❌ | ❌ |
| **AI Routing** | ✅ | ❌ | ❌ | ❌ |
| **Browser/WASM** | ✅ | ❌ | ❌ | ❌ |
| **Compression** | 2-32x | ❌ | ❌ | ❌ |
| **Open Source** | ✅ MIT | ❌ | ✅ | ✅ |

## Documentation

- [Getting Started Guide](https://github.com/ruvnet/ruvector/blob/main/docs/guide/GETTING_STARTED.md)
- [Cypher Reference](https://github.com/ruvnet/ruvector/blob/main/docs/api/CYPHER_REFERENCE.md)
- [GNN Architecture](https://github.com/ruvnet/ruvector/blob/main/docs/gnn-layer-implementation.md)
- [Performance Tuning](https://github.com/ruvnet/ruvector/blob/main/docs/optimization/PERFORMANCE_TUNING_GUIDE.md)
- [API Reference](https://github.com/ruvnet/ruvector/tree/main/docs/api)

## Contributing

```bash
# Clone repository
git clone https://github.com/ruvnet/ruvector.git
cd ruvector

# Install dependencies
npm install

# Run tests
npm test

# Build
npm run build
```

See [CONTRIBUTING.md](https://github.com/ruvnet/ruvector/blob/main/docs/development/CONTRIBUTING.md) for guidelines.

## License

MIT License — free for commercial and personal use.

---

<div align="center">

**Built by [rUv](https://ruv.io)** • [GitHub](https://github.com/ruvnet/ruvector) • [npm](https://npmjs.com/package/ruvector)

*Vector search that gets smarter over time.*

</div>
