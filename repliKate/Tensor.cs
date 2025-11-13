
namespace repliKate;

using System;
using System.Collections.Generic;
using System.Linq;

public class Tensor
{
    public int[] Shape { get; private set; }
    public float[] Data { get; private set; }
    public int Size 
    { 
        get => Data.Length;
        set { } // Setter for compatibility
    }

    // Constructor from size (1D tensor)
    public Tensor(int size)
    {
        Shape = new int[] { size };
        Data = new float[size];
    }

    // Constructor from shape
    public Tensor(int[] shape)
    {
        Shape = shape;
        int size = 1;
        foreach (int dim in shape)
            size *= dim;
        Data = new float[size];
    }

    // Constructor from data array (1D)
    public Tensor(float[] data)
    {
        Data = (float[])data.Clone();
        Shape = new int[] { data.Length };
    }

    public Tensor(float[] data, params int[] shape)
    {
        Shape = shape;
        int size = shape.Aggregate(1, (a, b) => a * b);
        if (data.Length != size)
            throw new ArgumentException("Data size doesn't match shape");
        Data = (float[])data.Clone();
    }

    public Tensor Clone()
    {
        return new Tensor((float[])Data.Clone(), Shape);
    }

    public float Magnitude()
    {
        float sum = 0;
        for (int i = 0; i < Size; i++)
            sum += Data[i] * Data[i];
        return (float)Math.Sqrt(sum);
    }

    public Tensor Normalize()
    {
        float mag = Magnitude();
        if (mag < 1e-8f) return Clone();
        return Scale(this, 1.0f / mag);
    }

    public static Tensor Scale(Tensor a, float scalar)
    {
        Tensor result = new Tensor(a.Shape);
        for (int i = 0; i < a.Size; i++)
            result.Data[i] = a.Data[i] * scalar;
        return result;
    }

    public void Randomize(Random random = null, float scale = 1.0f)
    {
        random = random ?? new Random();
        for (int i = 0; i < Size; i++)
        {
            Data[i] = ((float)random.NextDouble() * 2 - 1) * scale;
        }
    }

    public Dictionary<string, object> Serialize()
    {
        return new Dictionary<string, object>
        {
            { "shape", Shape },
            { "data", Data.ToList() }
        };
    }

    public static Tensor Deserialize(Dictionary<string, object> data)
    {
        var shape = ((List<object>)data["shape"]).Cast<int>().ToArray();
        var dataList = ((List<object>)data["data"]).Cast<float>().ToArray();
    
        Tensor tensor = new Tensor(shape);
        Array.Copy(dataList, tensor.Data, dataList.Length);
    
        return tensor;
    }
}
