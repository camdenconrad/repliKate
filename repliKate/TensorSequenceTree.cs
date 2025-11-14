using repliKate;

namespace repliKate;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

public class TensorNode
{
    public Tensor Value { get; private set; }
    private List<Tensor> nextTensors;
    private List<float> nextWeights;
    private float totalWeight;

    public TensorNode(Tensor value)
    {
        Value = value.Clone();
        nextTensors = new List<Tensor>();
        nextWeights = new List<float>();
        totalWeight = 0;
    }

    public void RecordNext(Tensor nextTensor, float weight = 1.0f)
    {
        int existingIndex = FindSimilarTensor(nextTensor, 0.95f);

        if (existingIndex >= 0)
        {
            nextWeights[existingIndex] += weight;
            float alpha = weight / nextWeights[existingIndex];

            for (int i = 0; i < nextTensors[existingIndex].Size; i++)
            {
                nextTensors[existingIndex].Data[i] =
                    alpha * nextTensor.Data[i] + (1 - alpha) * nextTensors[existingIndex].Data[i];
            }
        }
        else
        {
            nextTensors.Add(nextTensor.Clone());
            nextWeights.Add(weight);
        }

        totalWeight += weight;
    }

    private int FindSimilarTensor(Tensor query, float threshold)
    {
        for (int i = 0; i < nextTensors.Count; i++)
        {
            if (CosineSimilarity(query, nextTensors[i]) >= threshold)
                return i;
        }
        return -1;
    }

    public Tensor GetMostLikelyNext()
    {
        if (nextTensors.Count == 0) return null;

        int bestIndex = 0;
        float bestWeight = nextWeights[0];

        for (int i = 1; i < nextWeights.Count; i++)
        {
            if (nextWeights[i] > bestWeight)
            {
                bestWeight = nextWeights[i];
                bestIndex = i;
            }
        }

        return nextTensors[bestIndex].Clone();
    }

    public List<(Tensor tensor, float probability)> GetNextProbabilities()
    {
        if (totalWeight == 0) return new List<(Tensor, float)>();

        var result = new List<(Tensor, float)>();
        for (int i = 0; i < nextTensors.Count; i++)
        {
            result.Add((nextTensors[i].Clone(), nextWeights[i] / totalWeight));
        }

        return result.OrderByDescending(x => x.Item2).ToList();
    }

    public List<(Tensor tensor, float score)> GetTopNext(int count = 5)
    {
        if (nextTensors.Count == 0) return new List<(Tensor, float)>();
        if (totalWeight == 0) return new List<(Tensor, float)>();

        var scored = new List<(Tensor, float)>();
        for (int i = 0; i < nextTensors.Count; i++)
        {
            scored.Add((nextTensors[i].Clone(), nextWeights[i] / totalWeight));
        }

        return scored
            .OrderByDescending(x => x.Item2)
            .Take(count)
            .ToList();
    }

    public Tensor GetMostSimilarNext(Tensor queryTensor)
    {
        if (nextTensors.Count == 0) return null;

        int bestIndex = 0;
        float bestSimilarity = CosineSimilarity(queryTensor, nextTensors[0]);

        for (int i = 1; i < nextTensors.Count; i++)
        {
            float similarity = CosineSimilarity(queryTensor, nextTensors[i]);
            if (similarity > bestSimilarity)
            {
                bestSimilarity = similarity;
                bestIndex = i;
            }
        }

        return nextTensors[bestIndex].Clone();
    }

    public Tensor GetBlendedPrediction(int topK = 3)
    {
        if (nextTensors.Count == 0) return null;
        if (totalWeight == 0) return null;

        var topPredictions = GetTopNext(topK);
        if (topPredictions.Count == 0) return null;

        Tensor result = new Tensor(nextTensors[0].Shape);
        float sumWeights = topPredictions.Sum(p => p.score);

        foreach (var (tensor, score) in topPredictions)
        {
            float weight = score / sumWeights;
            for (int i = 0; i < result.Size; i++)
            {
                result.Data[i] += tensor.Data[i] * weight;
            }
        }

        return result;
    }

    private float CosineSimilarity(Tensor a, Tensor b)
    {
        if (a.Size != b.Size) return 0;

        float dot = 0;
        float magA = 0;
        float magB = 0;

        for (int i = 0; i < a.Size; i++)
        {
            dot += a.Data[i] * b.Data[i];
            magA += a.Data[i] * a.Data[i];
            magB += b.Data[i] * b.Data[i];
        }

        magA = (float)Math.Sqrt(magA);
        magB = (float)Math.Sqrt(magB);

        if (magA < 1e-8f || magB < 1e-8f) return 0;

        return dot / (magA * magB);
    }
}

public class TensorSequenceTree
{
    private List<TensorNode> nodes;
    private List<Tensor> fullSequence;
    private Dictionary<int, List<(List<Tensor> context, Dictionary<int, float> nextIndices)>> nGrams;
    private int maxContextWindow;
    private int tensorSize;
    private float similarityThreshold;

    private const int MAX_SEQUENCE_LENGTH = 50000;

    public TensorSequenceTree(int maxContextWindow = 100, float similarityThreshold = 0.95f)
    {
        nodes = new List<TensorNode>();
        fullSequence = new List<Tensor>();
        nGrams = new Dictionary<int, List<(List<Tensor>, Dictionary<int, float>)>>();

        this.maxContextWindow = Math.Max(2, Math.Min(maxContextWindow, 100));
        this.similarityThreshold = similarityThreshold;
        tensorSize = 0;

        for (int n = 2; n <= this.maxContextWindow; n++)
        {
            nGrams[n] = new List<(List<Tensor>, Dictionary<int, float>)>();
        }
    }

    public void SaveToFile(string filePath)
    {
        try
        {
            Console.WriteLine($"Starting compressed save to {filePath}...");
            var startTime = DateTime.Now;

            using (var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 65536))
            using (var gzipStream = new GZipStream(fileStream, CompressionLevel.Optimal))
            using (var writer = new StreamWriter(gzipStream, Encoding.UTF8, bufferSize: 65536))
            {
                writer.WriteLine("COMPRESSED_TENSOR_TREE_V2");
                writer.WriteLine($"CONTEXT_WINDOW:{maxContextWindow}");
                writer.WriteLine($"THRESHOLD:{similarityThreshold}");
                writer.WriteLine($"TENSOR_SIZE:{tensorSize}");
                writer.WriteLine($"NODE_COUNT:{nodes.Count}");
                writer.WriteLine($"SEQUENCE_LENGTH:{fullSequence.Count}");
                writer.WriteLine($"NGRAM_COUNTS:{string.Join(",", nGrams.Select(kvp => $"{kvp.Key}:{kvp.Value.Count}"))}");

                writer.WriteLine("===NODES_START===");
                for (int i = 0; i < nodes.Count; i++)
                {
                    if (i % 100 == 0)
                        Console.WriteLine($"Saving node {i}/{nodes.Count}...");

                    SaveNodeCompressed(writer, nodes[i]);
                }
                writer.WriteLine("===NODES_END===");

                writer.WriteLine("===NGRAMS_START===");
                SaveNGramsCompressed(writer);
                writer.WriteLine("===NGRAMS_END===");

                writer.WriteLine("===SEQUENCE_START===");
                int sequenceToSave = Math.Min(fullSequence.Count, 1000);
                writer.WriteLine($"SAVED_COUNT:{sequenceToSave}");
                for (int i = Math.Max(0, fullSequence.Count - sequenceToSave); i < fullSequence.Count; i++)
                {
                    SaveTensorCompressed(writer, fullSequence[i]);
                }
                writer.WriteLine("===SEQUENCE_END===");
            }

            var elapsed = (DateTime.Now - startTime).TotalSeconds;
            var fileInfo = new FileInfo(filePath);
            Console.WriteLine($"✅ Saved successfully in {elapsed:F2}s");
            Console.WriteLine($"   File size: {fileInfo.Length / 1024.0 / 1024.0:F2} MB");
            Console.WriteLine($"   Nodes: {nodes.Count}");
            Console.WriteLine($"   N-grams: {nGrams.Sum(kvp => kvp.Value.Count)}");
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to save TensorSequenceTree to {filePath}: {ex.Message}", ex);
        }
    }

    public static TensorSequenceTree LoadFromFile(string filePath)
    {
        try
        {
            Console.WriteLine($"Loading compressed model from {filePath}...");
            var startTime = DateTime.Now;

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Model file not found: {filePath}");
            }

            TensorSequenceTree tree = null;

            using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 65536))
            using (var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress))
            using (var reader = new StreamReader(gzipStream, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, bufferSize: 65536))
            {
                string version = reader.ReadLine();
                if (string.IsNullOrEmpty(version) || version != "COMPRESSED_TENSOR_TREE_V2")
                {
                    throw new InvalidDataException($"Unsupported file format. Expected 'COMPRESSED_TENSOR_TREE_V2', got '{version}'");
                }

                int contextWindow = int.Parse(reader.ReadLine()?.Split(':')[1] ?? "100");
                float threshold = float.Parse(reader.ReadLine()?.Split(':')[1] ?? "0.95");
                int tensorSize = int.Parse(reader.ReadLine()?.Split(':')[1] ?? "0");
                int nodeCount = int.Parse(reader.ReadLine()?.Split(':')[1] ?? "0");
                int sequenceLength = int.Parse(reader.ReadLine()?.Split(':')[1] ?? "0");
                string ngramCounts = reader.ReadLine() ?? "";

                Console.WriteLine($"   Context window: {contextWindow}");
                Console.WriteLine($"   Tensor size: {tensorSize}");
                Console.WriteLine($"   Nodes to load: {nodeCount}");

                tree = new TensorSequenceTree(contextWindow, threshold);
                tree.tensorSize = tensorSize;

                string line = reader.ReadLine();
                if (line != "===NODES_START===")
                {
                    throw new InvalidDataException($"Expected NODES_START marker, got: {line}");
                }

                for (int i = 0; i < nodeCount; i++)
                {
                    if (i % 100 == 0)
                        Console.WriteLine($"Loading node {i}/{nodeCount}...");

                    var node = LoadNodeCompressed(reader);
                    if (node != null)
                        tree.nodes.Add(node);
                }

                line = reader.ReadLine();
                if (line != "===NODES_END===")
                {
                    throw new InvalidDataException($"Expected NODES_END marker, got: {line}");
                }

                line = reader.ReadLine();
                if (line != "===NGRAMS_START===")
                {
                    throw new InvalidDataException($"Expected NGRAMS_START marker, got: {line}");
                }

                tree.LoadNGramsCompressed(reader);

                // The LoadNGramsCompressed method reads until it hits ===NGRAMS_END=== internally
                // So we need to check if we're at SEQUENCE_START
                line = reader.ReadLine();
                if (line == "===SEQUENCE_START===")
                {
                    string countLine = reader.ReadLine();
                    if (countLine != null && countLine.StartsWith("SAVED_COUNT:"))
                    {
                        int savedCount = int.Parse(countLine.Split(':')[1]);
                        Console.WriteLine($"Loading {savedCount} reference tensors...");

                        for (int i = 0; i < savedCount; i++)
                        {
                            var tensor = LoadTensorCompressed(reader);
                            if (tensor != null)
                                tree.fullSequence.Add(tensor);
                        }

                        line = reader.ReadLine();
                        if (line != "===SEQUENCE_END===")
                        {
                            Console.WriteLine($"Warning: Expected SEQUENCE_END marker, got: {line}");
                        }
                    }
                }
            }

            var elapsed = (DateTime.Now - startTime).TotalSeconds;
            Console.WriteLine($"✅ Loaded successfully in {elapsed:F2}s");
            Console.WriteLine($"   Loaded {tree.nodes.Count} nodes");
            Console.WriteLine($"   Loaded {tree.fullSequence.Count} sequence tensors");
            return tree;
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to load TensorSequenceTree from {filePath}: {ex.Message}", ex);
        }
    }

    private void SaveNodeCompressed(StreamWriter writer, TensorNode node)
    {
        writer.WriteLine("NODE_START");
        SaveTensorCompressed(writer, node.Value);

        var nextProbs = node.GetNextProbabilities();
        writer.WriteLine($"NEXT_COUNT:{nextProbs.Count}");

        foreach (var (tensor, prob) in nextProbs)
        {
            writer.WriteLine($"PROB:{prob:F6}");
            SaveTensorCompressed(writer, tensor);
        }

        writer.WriteLine("NODE_END");
    }

    private static TensorNode LoadNodeCompressed(StreamReader reader)
    {
        try
        {
            string line = reader.ReadLine();
            if (line != "NODE_START")
            {
                throw new InvalidDataException($"Expected NODE_START, got: {line}");
            }

            Tensor value = LoadTensorCompressed(reader);
            if (value == null)
            {
                throw new InvalidDataException("Failed to load node value tensor");
            }

            TensorNode node = new TensorNode(value);

            line = reader.ReadLine();
            if (line == null || !line.StartsWith("NEXT_COUNT:"))
            {
                throw new InvalidDataException($"Expected NEXT_COUNT, got: {line}");
            }

            int nextCount = int.Parse(line.Split(':')[1]);

            for (int i = 0; i < nextCount; i++)
            {
                line = reader.ReadLine();
                if (line == null || !line.StartsWith("PROB:"))
                {
                    throw new InvalidDataException($"Expected PROB, got: {line}");
                }

                float prob = float.Parse(line.Split(':')[1]);
                Tensor nextTensor = LoadTensorCompressed(reader);
                if (nextTensor != null)
                {
                    node.RecordNext(nextTensor, prob);
                }
            }

            line = reader.ReadLine();
            if (line != "NODE_END")
            {
                throw new InvalidDataException($"Expected NODE_END, got: {line}");
            }

            return node;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading node: {ex.Message}");
            return null;
        }
    }

    private void SaveTensorCompressed(StreamWriter writer, Tensor tensor)
    {
        writer.WriteLine($"T:{tensor.Size}:{string.Join(",", tensor.Shape)}");

        const int CHUNK_SIZE = 100;
        for (int i = 0; i < tensor.Data.Length; i += CHUNK_SIZE)
        {
            int count = Math.Min(CHUNK_SIZE, tensor.Data.Length - i);
            var chunk = tensor.Data.Skip(i).Take(count).Select(f => f.ToString("F6"));
            writer.WriteLine($"D:{string.Join(",", chunk)}");
        }
    }

    private static Tensor LoadTensorCompressed(StreamReader reader)
    {
        try
        {
            string line = reader.ReadLine();
            if (line == null || !line.StartsWith("T:"))
            {
                throw new InvalidDataException($"Expected tensor header (T:), got: {line}");
            }

            var parts = line.Substring(2).Split(':');
            int size = int.Parse(parts[0]);
            int[] shape = parts[1].Split(',').Select(int.Parse).ToArray();

            Tensor tensor = new Tensor(shape);
            int dataIndex = 0;

            while (dataIndex < size)
            {
                line = reader.ReadLine();
                if (line == null || !line.StartsWith("D:"))
                    break;

                var values = line.Substring(2).Split(',').Select(float.Parse).ToArray();
                Array.Copy(values, 0, tensor.Data, dataIndex, values.Length);
                dataIndex += values.Length;
            }

            if (dataIndex != size)
            {
                Console.WriteLine($"Warning: Expected {size} values, got {dataIndex}");
            }

            return tensor;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading tensor: {ex.Message}");
            return null;
        }
    }

    private void SaveNGramsCompressed(StreamWriter writer)
    {
        foreach (var kvp in nGrams)
        {
            int n = kvp.Key;
            var entries = kvp.Value;

            if (entries.Count == 0) continue;

            writer.WriteLine($"NGRAM:{n}:{entries.Count}");

            int saved = 0;
            foreach (var entry in entries)
            {
                if (saved % 100 == 0 && saved > 0)
                    Console.WriteLine($"Saving {n}-gram {saved}/{entries.Count}...");

                writer.WriteLine($"CONTEXT:{entry.context.Count}");
                foreach (var tensor in entry.context)
                {
                    SaveTensorCompressed(writer, tensor);
                }

                writer.WriteLine($"INDICES:{entry.nextIndices.Count}");
                foreach (var idx in entry.nextIndices)
                {
                    writer.WriteLine($"IDX:{idx.Key}:{idx.Value:F6}");
                }

                saved++;
            }
        }
        writer.WriteLine("NGRAM_COMPLETE");
    }

    private void LoadNGramsCompressed(StreamReader reader)
    {
        try
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (line == "===NGRAMS_END===")
                {
                    break;
                }

                if (line == "NGRAM_COMPLETE")
                {
                    // Read the ===NGRAMS_END=== marker
                    reader.ReadLine();
                    break;
                }

                if (!line.StartsWith("NGRAM:"))
                    continue;

                var parts = line.Split(':');
                int n = int.Parse(parts[1]);
                int count = int.Parse(parts[2]);

                Console.WriteLine($"Loading {count} {n}-grams...");

                if (!nGrams.ContainsKey(n))
                    nGrams[n] = new List<(List<Tensor>, Dictionary<int, float>)>();

                for (int i = 0; i < count; i++)
                {
                    if (i % 100 == 0 && i > 0)
                        Console.WriteLine($"Loading {n}-gram {i}/{count}...");

                    line = reader.ReadLine();
                    if (line == null || !line.StartsWith("CONTEXT:"))
                    {
                        Console.WriteLine($"Warning: Expected CONTEXT, got: {line}");
                        continue;
                    }

                    int contextCount = int.Parse(line.Split(':')[1]);
                    var context = new List<Tensor>();
                    for (int j = 0; j < contextCount; j++)
                    {
                        var tensor = LoadTensorCompressed(reader);
                        if (tensor != null)
                            context.Add(tensor);
                    }

                    line = reader.ReadLine();
                    if (line == null || !line.StartsWith("INDICES:"))
                    {
                        Console.WriteLine($"Warning: Expected INDICES, got: {line}");
                        continue;
                    }

                    int indicesCount = int.Parse(line.Split(':')[1]);
                    var indices = new Dictionary<int, float>();
                    for (int j = 0; j < indicesCount; j++)
                    {
                        line = reader.ReadLine();
                        if (line != null && line.StartsWith("IDX:"))
                        {
                            var idxParts = line.Split(':');
                            int idx = int.Parse(idxParts[1]);
                            float weight = float.Parse(idxParts[2]);
                            indices[idx] = weight;
                        }
                    }

                    if (context.Count > 0)
                    {
                        nGrams[n].Add((context, indices));
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading n-grams: {ex.Message}");
        }
    }

    public int GetMaxContextWindow() => maxContextWindow;
    public int GetTensorSize() => tensorSize;

    public void Learn(Tensor[] sequence)
    {
        if (sequence == null || sequence.Length == 0) return;

        if (tensorSize == 0 && sequence.Length > 0)
        {
            tensorSize = sequence[0].Size;
        }

        foreach (var tensor in sequence)
        {
            if (tensor.Size != tensorSize)
                throw new ArgumentException($"All tensors must have size {tensorSize}");
        }

        // Append the sequence and compute the base index of these new items in the global fullSequence
        fullSequence.AddRange(sequence.Select(t => t.Clone()));
        int baseIndex = Math.Max(0, fullSequence.Count - sequence.Length);

        if (fullSequence.Count > MAX_SEQUENCE_LENGTH)
        {
            int toRemove = fullSequence.Count - MAX_SEQUENCE_LENGTH;
            fullSequence.RemoveRange(0, toRemove);
            Console.WriteLine($"⚠️ Trimmed sequence to {MAX_SEQUENCE_LENGTH} tensors (removed oldest {toRemove})");
            // After trimming, the baseIndex might shift, but since we just added the latest items,
            // their indices remain at the end and thus still valid.
        }

        for (int i = 0; i < sequence.Length - 1; i++)
        {
            Tensor current = sequence[i];
            Tensor next = sequence[i + 1];

            int nodeIndex = FindOrCreateNode(current);
            nodes[nodeIndex].RecordNext(next);
        }

        FindOrCreateNode(sequence[sequence.Length - 1]);
        BuildNGrams(sequence, baseIndex);
    }

    private int FindOrCreateNode(Tensor tensor)
    {
        for (int i = 0; i < nodes.Count; i++)
        {
            if (CosineSimilarity(tensor, nodes[i].Value) >= similarityThreshold)
            {
                float alpha = 0.2f;
                for (int j = 0; j < tensor.Size; j++)
                {
                    nodes[i].Value.Data[j] = alpha * tensor.Data[j] + (1 - alpha) * nodes[i].Value.Data[j];
                }
                return i;
            }
        }

        nodes.Add(new TensorNode(tensor));
        return nodes.Count - 1;
    }

    private void BuildNGrams(Tensor[] sequence, int baseIndex)
    {
        for (int n = 2; n <= maxContextWindow; n++)
        {
            if (sequence.Length < n)
                continue;

            for (int i = 0; i <= sequence.Length - n; i++)
            {
                List<Tensor> context = new List<Tensor>();
                for (int j = 0; j < n - 1; j++)
                {
                    context.Add(sequence[i + j].Clone());
                }

                // Map the local next index within this sequence to the global fullSequence index
                int nextIndex = baseIndex + i + n - 1;
                AddNGram(n, context, nextIndex);
            }
        }
    }

    private void AddNGram(int n, List<Tensor> context, int nextIndex)
    {
        bool found = false;
        foreach (var entry in nGrams[n])
        {
            if (ContextMatches(entry.context, context))
            {
                if (!entry.nextIndices.ContainsKey(nextIndex))
                    entry.nextIndices[nextIndex] = 0;
                entry.nextIndices[nextIndex]++;
                found = true;
                break;
            }
        }

        if (!found)
        {
            var newEntry = (new List<Tensor>(context.Select(t => t.Clone())), new Dictionary<int, float> { { nextIndex, 1.0f } });
            nGrams[n].Add(newEntry);
        }
    }

    private bool ContextMatches(List<Tensor> context1, List<Tensor> context2)
    {
        if (context1.Count != context2.Count) return false;

        for (int i = 0; i < context1.Count; i++)
        {
            if (CosineSimilarity(context1[i], context2[i]) < similarityThreshold)
                return false;
        }

        return true;
    }

    public Tensor[] PredictNext(Tensor[] context, int count = 1, bool useBlending = false)
    {
        if (context == null || context.Length == 0)
            return new Tensor[0];

        List<Tensor> predictions = new List<Tensor>();
        List<Tensor> currentContext = new List<Tensor>(context.Select(t => t.Clone()));

        for (int i = 0; i < count; i++)
        {
            Tensor next = PredictSingleNext(currentContext.ToArray(), useBlending);

            if (next == null) break;

            predictions.Add(next);
            currentContext.Add(next);

            if (currentContext.Count > maxContextWindow)
            {
                currentContext.RemoveAt(0);
            }
        }

        return predictions.ToArray();
    }

    public List<(Tensor tensor, float confidence)> GetTopPredictions(Tensor[] context, int topN = 5)
    {
        if (context == null || context.Length == 0)
            return new List<(Tensor, float)>();

        var candidates = new Dictionary<int, float>();

        for (int n = maxContextWindow; n >= 2; n--)
        {
            if (nGrams.ContainsKey(n) && context.Length >= n - 1)
            {
                float weight = (float)n;
                TryAddPredictions(candidates, context, n, weight);
            }
        }

        if (context.Length > 0)
        {
            Tensor lastTensor = context[context.Length - 1];
            int nodeIndex = FindSimilarNode(lastTensor);

            if (nodeIndex >= 0)
            {
                var topNext = nodes[nodeIndex].GetTopNext(topN);
                foreach (var (tensor, score) in topNext)
                {
                    int index = FindTensorIndex(tensor);
                    if (index >= 0)
                    {
                        if (!candidates.ContainsKey(index))
                            candidates[index] = 0;
                        candidates[index] += score * 1.0f;
                    }
                }
            }
        }

        float total = candidates.Values.Sum();
        if (total == 0) return new List<(Tensor, float)>();

        return candidates
            .OrderByDescending(kvp => kvp.Value)
            .Where(kvp => kvp.Key >= 0 && kvp.Key < fullSequence.Count)
            .Take(topN)
            .Select(kvp => (fullSequence[kvp.Key].Clone(), kvp.Value / total))
            .ToList();
    }

    private void TryAddPredictions(Dictionary<int, float> candidates, Tensor[] context, int n, float weight)
    {
        int contextSize = n - 1;
        if (context.Length < contextSize) return;

        List<Tensor> contextWindow = context
            .Skip(context.Length - contextSize)
            .Take(contextSize)
            .ToList();

        foreach (var entry in nGrams[n])
        {
            if (ContextMatches(entry.context, contextWindow))
            {
                float total = entry.nextIndices.Values.Sum();
                foreach (var kvp in entry.nextIndices)
                {
                    if (!candidates.ContainsKey(kvp.Key))
                        candidates[kvp.Key] = 0;

                    float probability = kvp.Value / total;
                    candidates[kvp.Key] += probability * weight;
                }
                break;
            }
        }
    }

    private Tensor PredictSingleNext(Tensor[] context, bool useBlending)
    {
        var topPredictions = GetTopPredictions(context, useBlending ? 3 : 1);

        if (topPredictions.Count == 0)
            return null;

        if (!useBlending)
            return topPredictions[0].tensor;

        Tensor result = new Tensor(tensorSize);
        float totalConfidence = topPredictions.Sum(p => p.confidence);

        foreach (var (tensor, confidence) in topPredictions)
        {
            float weight = confidence / totalConfidence;
            for (int i = 0; i < result.Size; i++)
            {
                result.Data[i] += tensor.Data[i] * weight;
            }
        }

        return result;
    }

    private int FindSimilarNode(Tensor tensor)
    {
        for (int i = 0; i < nodes.Count; i++)
        {
            if (CosineSimilarity(tensor, nodes[i].Value) >= similarityThreshold)
                return i;
        }
        return -1;
    }

    private int FindTensorIndex(Tensor tensor)
    {
        for (int i = 0; i < fullSequence.Count; i++)
        {
            if (CosineSimilarity(tensor, fullSequence[i]) >= 0.99f)
                return i;
        }
        return -1;
    }

    public Tensor[] ContinueSequence(int count = 10, bool useBlending = false)
    {
        if (fullSequence.Count == 0)
            return new Tensor[0];

        int contextSize = Math.Min(maxContextWindow - 1, fullSequence.Count);
        Tensor[] context = fullSequence
            .Skip(fullSequence.Count - contextSize)
            .Take(contextSize)
            .ToArray();

        return PredictNext(context, count, useBlending);
    }

    public List<(Tensor tensor, float similarity)> GetSimilarTensors(Tensor queryTensor, int topN = 5)
    {
        var similarities = new List<(Tensor, float)>();

        foreach (var tensor in fullSequence)
        {
            float similarity = CosineSimilarity(queryTensor, tensor);
            similarities.Add((tensor.Clone(), similarity));
        }

        return similarities
            .OrderByDescending(s => s.Item2)
            .Take(topN)
            .ToList();
    }

    public Tensor Interpolate(Tensor from, Tensor to, float t)
    {
        if (from.Size != to.Size)
            throw new ArgumentException("Tensors must have same size");

        Tensor result = new Tensor(from.Size);
        for (int i = 0; i < from.Size; i++)
        {
            result.Data[i] = from.Data[i] * (1 - t) + to.Data[i] * t;
        }

        return result;
    }

    public string GetStatistics()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine($"Learned sequence length: {fullSequence.Count} tensors");
        sb.AppendLine($"Unique tensor nodes: {nodes.Count}");
        sb.AppendLine($"Tensor size: {tensorSize}");
        sb.AppendLine($"Max context window: {maxContextWindow}");
        sb.AppendLine($"Similarity threshold: {similarityThreshold:F3}");

        for (int n = 2; n <= maxContextWindow; n++)
        {
            if (nGrams.ContainsKey(n))
            {
                string gramName = n == 2 ? "Bigrams" :
                                  n == 3 ? "Trigrams" :
                                  $"{n}-grams";
                sb.AppendLine($"{gramName}: {nGrams[n].Count}");
            }
        }

        if (nodes.Count > 0)
        {
            float avgTransitions = (float)nodes.Average(n => n.GetNextProbabilities().Count);
            sb.AppendLine($"Average transitions per node: {avgTransitions:F2}");
        }

        return sb.ToString();
    }

    private float CosineSimilarity(Tensor a, Tensor b)
    {
        if (a.Size != b.Size) return 0;

        float dot = 0;
        float magA = 0;
        float magB = 0;

        for (int i = 0; i < a.Size; i++)
        {
            dot += a.Data[i] * b.Data[i];
            magA += a.Data[i] * a.Data[i];
            magB += b.Data[i] * b.Data[i];
        }

        magA = (float)Math.Sqrt(magA);
        magB = (float)Math.Sqrt(magB);

        if (magA < 1e-8f || magB < 1e-8f) return 0;

        return dot / (magA * magB);
    }

    public void Clear()
    {
        nodes.Clear();
        fullSequence.Clear();
        foreach (var dict in nGrams.Values)
        {
            dict.Clear();
        }
        tensorSize = 0;
    }
}