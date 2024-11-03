from vidur.types.base_int_enum import BaseIntEnum

class StorageComputeType(BaseIntEnum):
    CAN_COMPUTE = 1
    ONLY_STORAGE = 2

class StorageDeviceType(BaseIntEnum):
    GPU = 0
    CPU = 1
    DISK = 2
