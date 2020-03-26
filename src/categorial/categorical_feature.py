from sklearn import preprocessing

"""
 - label encoding
 - one hot encoding
 - birarization

"""


class CategoricalFeatures:
    def __init__(self, dataframe, categorical_features, encoding_type, handle_na=False):
        """
        :param df: pandas df
        :param categorical_features: list of column names e.g. ["ord_1", "nom_0", ....]
        :param encoding_type: binary, label

        """
        self.df = dataframe
        self.categorical_features = categorical_features
        self.encoding_type = encoding_type
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.one_hot_encoders = dict()
        self.handle_na = handle_na

        if handle_na:
            for c in self.categorical_features:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("999999")

        self.output_df = self.df.copy(deep=True)  # if something changed in original df, output_df will not change

    def _label_encoding(self):
        for catfeat in self.categorical_features:
            lbl = preprocessing.LabelEncoder()
            # fit in the original df
            lbl.fit(self.df[catfeat].values)
            self.output_df.loc[:, catfeat] = lbl.transform(self.df[catfeat].values)
            self.label_encoders[catfeat] = lbl
        return self.output_df

    def _label_binarization(self):
        for catfeat in self.categorical_features:
            lbl = preprocessing.LabelBinarizer()
            # fit in the original df
            lbl.fit(self.df[catfeat].values)
            val = lbl.transform(self.df[catfeat].values)  # array
            self.output_df = self.output_df.drop(catfeat, axis=1)
            for j in range(val.shape[1]):
                new_col_name = catfeat + f"__bin__{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[catfeat] = lbl
        return self.output_df

    def _one_hot_encoding(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.categorical_features].values)
        return ohe.transform(self.output_df[self.categorical_features].values)

    def fit_transform(self):
        if self.encoding_type == "label":
            return self._label_encoding()
        elif self.encoding_type == "binary":
            return self._label_binarization()
        elif self.encoding_type == "ohe":
            return self._one_hot_encoding()

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.categorical_features:
                dataframe.loc[:, c] = dataframe[:, c].astype(str).fillna("-999999")
        if self.encoding_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        elif self.encoding_type == "binary":
            for c, lbl in self._label_binarization().items():
                val = lbl.transform(dataframe[c].values)  # an array
                dataframe = dataframe.drop(c, axis=1)
                for j in range(val.shape[1]):
                    new_col_name = f"__bin__{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe
        elif self.encoding_type == "ohe":
            return self.one_hot_encoders(dataframe[self.categorical_features].values)
        else:
            raise Exception("Encoding type not matched")


if __name__ == '__main__':
    import pandas as pd
    from sklearn import linear_model

    df = pd.read_csv("input/train.csv")
    df_test = pd.read_csv("input/test.csv")
    sample = pd.read_csv("input/sample_submission.csv")

    train_len = len(df)
    df_test["target"] = -1
    full_data = pd.concat([df, df_test])

    cols = [col for col in df.columns if col not in ['id', 'target']]
    cat_feats = CategoricalFeatures(dataframe=full_data,
                                    categorical_features=cols,
                                    encoding_type="ohe",
                                    handle_na=True)

    full_data_transformed = cat_feats.fit_transform()
    X = full_data_transformed[:train_len, :]
    X_test = full_data_transformed[train_len:, :]

    clf = linear_model.LogisticRegression()
    clf.fit(X, df.target.values)
    preds = clf.predict_proba(X_test)[:, 1]

    sample.loc[:, "target"] = preds
    sample.to_csv("output/submissions.csv", index=False)
