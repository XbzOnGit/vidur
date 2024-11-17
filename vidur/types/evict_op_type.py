from vidur.types.base_int_enum import BaseIntEnum


class EvictOpType(BaseIntEnum):
    NONE = 0
    WRITE_TO_LOWER = 1
    DROP = 2
    COMPRESS = 3
    SWAP = 4
