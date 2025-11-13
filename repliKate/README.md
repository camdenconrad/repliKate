# repliKate

A small, dependency-free .NET 9 class library for recording, compressing, and predicting sequences of float tensors. Extracted from a chess project, but now game-agnostic. Use it to learn patterns in any domain: games, signals, user actions, etc.

## Features
- Tensor value type with shape and operations (clone, normalize, randomize, scale)
- Sequence model `TensorSequenceTree` that:
  - Records observed next tensors with weights
  - Merges very similar tensors via cosine similarity
  - Retrieves most likely or top-N next tensors with scores
  - Supports simple similarity search and sampling
  - Binary and zip serialization utilities for persistence

## Install
This repository includes the library project.

```bash
dotnet build repliKate/repliKate.csproj
```

Add a project reference from your app or library, or publish this as a package in your own feed.

## Usage

```csharp
using repliKate;

// Create a sequence tree
var tree = new TensorSequenceTree(similarityThreshold: 0.95f);

// Record sequences of tensors (your domain encodes to float arrays)
var a = new Tensor(new float[] { 1, 0, 0 });
var b = new Tensor(new float[] { 0, 1, 0 });
var c = new Tensor(new float[] { 0, 0, 1 });

tree.RecordSequence(new[] { a, b, c });

// Query most likely next after 'b'
var nodeB = tree.GetOrCreateNode(b);
var next = nodeB.GetMostLikelyNext();

// Get top 3 next with scores
var top3 = nodeB.GetTopNext(3);

// Save and load
using var fs = File.Create("model.bin");
tree.SaveBinary(fs);

fs.Position = 0;
var loaded = TensorSequenceTree.LoadBinary(fs);
```

## Namespaces
- Library root namespace: `repliKate`
- Types: `Tensor`, `TensorSequenceTree`, and internal `TensorNode`

## What was removed
- All chess-specific logic, UI, and Stockfish integration
- Avalonia dependencies and application entry points

## Repository layout
- `repliKate/` — library source (only `Tensor.cs`, `TensorSequenceTree.cs` compiled)
- Other files remain in the repo history for reference but are not part of the library build

## License
MIT (or the license of the parent repository if different).
