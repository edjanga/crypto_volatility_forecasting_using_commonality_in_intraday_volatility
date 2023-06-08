class TensorDecomposition:

    def __init__(self, rv: pd.DataFrame):
        self._rv = rv
        self._rv.index = \
            pd.MultiIndex.from_tuples([(date, time) for date, time in zip(self._rv.index.date, self._rv.index.time)])
        self._rv.index = self._rv.index.set_names(['Date', 'Time'])
        self._rv = self._rv.stack()
        self._rv.index = self._rv.index.set_names(['Date', 'Time', 'Symbol'])
        self.tensor = pd_to_tensor(self._rv, keep_index=True)
        pdb.set_trace()