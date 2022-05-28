selectors = {
    "kitti": {
        "sample_stride": 1,
        "train": lambda length: [i for i in range(0, min(100, length), 1) if i % 10],
        "val": lambda length: [i for i in range(0, min(100, length), 10)],
        "test": lambda length: [i for i in range(0, min(100, length), 1)],
    },
}
