from __future__ import print_function, division


class TripletDataset:
    def __init__(self, triplet_indices=None, pre_images=None, transform=None):
        self.triplet_indices = triplet_indices
        self.pre_images = pre_images
        self.transform = transform
    def __getitem__(self, index):
        index1 = self.triplet_indices["A"][index]
        index2 = self.triplet_indices["B"][index]
        index3 = self.triplet_indices["C"][index]

        anchor = self.pre_images[int(index1)]
        positive = self.pre_images[int(index2)]
        negative = self.pre_images[int(index3)]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return (anchor, positive, negative)

    def __len__(self):
        return len(self.triplet_indices)