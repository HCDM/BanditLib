import pickle
import numpy as np
import os

FOLDDATA_WRITE_VERSION = 3


class DataSet(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.FeatureMap = {}
        self.FeatureMatrix = None
        self.DoclistRanges = None
        self.LabelVector = None

    def _read_file(self, path):
        CurrentQid = None
        queries = {}
        queryIndex = 0
        doclists = []
        labels = []
        AllFeatures = {}
        FeatureMax = {}
        FeatureMin = {}

        for line in open(path, "r"):
            info = line[: line.find("#")].split()
            qid = info[1].split(":")[1]
            label = int(info[0])
            if qid not in queries:
                queryIndex = len(queries)
                queries[qid] = queryIndex
                doclists.append([])
                labels.append([])
                CurrentQid = qid
            elif qid != CurrentQid:
                queryIndex = queries[qid]
                CurrentQid = qid

            FeatureDict = {}
            for pair in info[2:]:
                FeatureID, FeatureValue = pair.split(":")
                # FeatureID = int(FeatureID)
                AllFeatures[FeatureID] = True
                FeatureValue = float(FeatureValue)
                FeatureDict[FeatureID] = FeatureValue
                if FeatureID in FeatureMax:
                    FeatureMax[FeatureID] = max(FeatureMax[FeatureID], FeatureValue)
                    FeatureMin[FeatureID] = min(FeatureMin[FeatureID], FeatureValue)
                else:
                    FeatureMax[FeatureID] = FeatureValue
                    FeatureMin[FeatureID] = FeatureValue

            doclists[queryIndex].append(FeatureDict)
            labels[queryIndex].append(label)
        return queries, doclists, labels, AllFeatures

    def create_feature_mapping(self, FeatureDict):
        TotalFeatures = len(self.FeatureMap)
        # FeatureMap = {}
        for fid in FeatureDict:
            if fid not in self.FeatureMap:
                self.FeatureMap[fid] = TotalFeatures
                TotalFeatures += 1

    def convert_FeatureDict(self, Doclists, LabelLists, FeatureMapping, query_level_norm=True):
        TotalFeatures = len(FeatureMapping)
        TotalDocs = 0
        Ranges = []
        for doclist in Doclists:
            StartIdx = TotalDocs
            TotalDocs += len(doclist)
            Ranges.append((StartIdx, TotalDocs))

        FeatureMatrix = np.zeros((TotalFeatures, TotalDocs))
        LabelVector = np.zeros(TotalDocs, dtype=np.int32)

        index = 0
        for doclist, labels in zip(Doclists, LabelLists):
            start_index = index
            for FeatureDict, label in zip(doclist, labels):
                for idx, value in FeatureDict.items():
                    if idx in FeatureMapping:
                        FeatureMatrix[FeatureMapping[idx], index] = value
                LabelVector[index] = label
                index += 1
            end_index = index
            if query_level_norm:
                FeatureMatrix[:, start_index:end_index] -= np.amin(
                    FeatureMatrix[:, start_index:end_index], axis=1
                )[:, None]
                safe_max = np.amax(FeatureMatrix[:, start_index:end_index], axis=1)
                safe_ind = safe_max != 0
                FeatureMatrix[safe_ind, start_index:end_index] /= safe_max[safe_ind][:, None]

        QueryPointer = np.zeros(len(Ranges) + 1, dtype=np.int32)
        for i, ra in enumerate(Ranges):
            QueryPointer[i + 1] = ra[1]

        return FeatureMatrix, QueryPointer, LabelVector

    def read_data(self):
        data_pickle_path = self.data_path + "binarized_data.npz"
        fmap_pickle_path = self.data_path + "binarized_fmap.pickle"

        fmap_read = False
        data_read = False
        if os.path.isfile(fmap_pickle_path):
            with open(fmap_pickle_path, "rb") as f:
                loaded = pickle.load(f)
                self.FeatureMap = loaded[1]
                fmap_read = True
        if os.path.isfile(data_pickle_path):
            data = np.load(data_pickle_path)
            self.FeatureMatrix = data["feature_matrix"]
            self.DoclistRanges = data["doclist_ranges"]
            self.LabelVector = data["label_vector"]
            data_read = True

        if not fmap_read or not data_read:
            doclists = []
            labels = []

            for name in ["train.txt", "vali.txt", "test.txt"]:
                _, n_doclists, n_labels, features = self._read_file(self.data_path + name)
                doclists.extend(n_doclists)
                labels.extend(n_labels)
                self.create_feature_mapping(features)

            with open(fmap_pickle_path, "wb") as f:
                pickle.dump((FOLDDATA_WRITE_VERSION, self.FeatureMap), f)

            self.FeatureMatrix, self.DoclistRanges, self.LabelVector = self.convert_FeatureDict(
                doclists, labels, self.FeatureMap
            )

            np.savez(
                data_pickle_path,
                feature_map=self.FeatureMap,
                feature_matrix=self.FeatureMatrix,
                doclist_ranges=self.DoclistRanges,
                label_vector=self.LabelVector,
            )

        self.num_features = self.FeatureMatrix.shape[0]
        self.FeatureMatrix = self.FeatureMatrix.T
        self.n_queries = self.DoclistRanges.shape[0] - 1
        self.n_docs = self.FeatureMatrix.shape[0]
