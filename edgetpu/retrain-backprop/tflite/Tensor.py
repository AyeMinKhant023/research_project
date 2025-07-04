# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Tensor(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Tensor()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensor(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def TensorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Tensor
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Tensor
    def Shape(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Tensor
    def ShapeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Tensor
    def ShapeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Tensor
    def ShapeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # Tensor
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Tensor
    def Buffer(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Tensor
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Tensor
    def Quantization(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from tflite.QuantizationParameters import QuantizationParameters
            obj = QuantizationParameters()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Tensor
    def IsVariable(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # Tensor
    def Sparsity(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from tflite.SparsityParameters import SparsityParameters
            obj = SparsityParameters()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Tensor
    def ShapeSignature(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Tensor
    def ShapeSignatureAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Tensor
    def ShapeSignatureLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Tensor
    def ShapeSignatureIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # Tensor
    def HasRank(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # Tensor
    def VariantTensors(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.VariantSubType import VariantSubType
            obj = VariantSubType()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Tensor
    def VariantTensorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Tensor
    def VariantTensorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

def TensorStart(builder): builder.StartObject(10)
def Start(builder):
    return TensorStart(builder)
def TensorAddShape(builder, shape): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(shape), 0)
def AddShape(builder, shape):
    return TensorAddShape(builder, shape)
def TensorStartShapeVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartShapeVector(builder, numElems):
    return TensorStartShapeVector(builder, numElems)
def TensorAddType(builder, type): builder.PrependInt8Slot(1, type, 0)
def AddType(builder, type):
    return TensorAddType(builder, type)
def TensorAddBuffer(builder, buffer): builder.PrependUint32Slot(2, buffer, 0)
def AddBuffer(builder, buffer):
    return TensorAddBuffer(builder, buffer)
def TensorAddName(builder, name): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def AddName(builder, name):
    return TensorAddName(builder, name)
def TensorAddQuantization(builder, quantization): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(quantization), 0)
def AddQuantization(builder, quantization):
    return TensorAddQuantization(builder, quantization)
def TensorAddIsVariable(builder, isVariable): builder.PrependBoolSlot(5, isVariable, 0)
def AddIsVariable(builder, isVariable):
    return TensorAddIsVariable(builder, isVariable)
def TensorAddSparsity(builder, sparsity): builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(sparsity), 0)
def AddSparsity(builder, sparsity):
    return TensorAddSparsity(builder, sparsity)
def TensorAddShapeSignature(builder, shapeSignature): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(shapeSignature), 0)
def AddShapeSignature(builder, shapeSignature):
    return TensorAddShapeSignature(builder, shapeSignature)
def TensorStartShapeSignatureVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartShapeSignatureVector(builder, numElems):
    return TensorStartShapeSignatureVector(builder, numElems)
def TensorAddHasRank(builder, hasRank): builder.PrependBoolSlot(8, hasRank, 0)
def AddHasRank(builder, hasRank):
    return TensorAddHasRank(builder, hasRank)
def TensorAddVariantTensors(builder, variantTensors): builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(variantTensors), 0)
def AddVariantTensors(builder, variantTensors):
    return TensorAddVariantTensors(builder, variantTensors)
def TensorStartVariantTensorsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartVariantTensorsVector(builder, numElems):
    return TensorStartVariantTensorsVector(builder, numElems)
def TensorEnd(builder): return builder.EndObject()
def End(builder):
    return TensorEnd(builder)