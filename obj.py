class Chunk(object):
    def __init__(self, chunk_hash="", index=0, keeper="") -> None:
        self.chunk_hash = chunk_hash
        self.index = index
        self.keeper = keeper

    def serialize(self, ck):
        ck.hash = self.chunk_hash
        ck.index = self.index
        ck.keeper = self.keeper

    def deserialize(self, ck):
        self.chunk_hash = ck.hash
        self.index = ck.index
        self.keeper = ck.keeper

    def to_json(self):
        json_obj = {"hash": self.chunk_hash, "index": self.index, "keeper": self.keeper}
        return json_obj


class Model(object):
    def __init__(self, owner="", rd=1, model_hash="", scores=None, chunks=None) -> None:
        self.owner = owner
        self.rd = rd
        self.model_hash = model_hash
        self.scores = scores
        self.chunks = chunks

    def serialize(self, md):
        md.owner = self.owner
        md.rd = self.rd
        md.model_hash = self.model_hash
        md.scores.update(self.scores)
        for c in self.chunks:
            ck = md.chunks.add()
            c.serialize(ck)

    def deserialize(self, md):
        self.owner = md.owner
        self.rd = md.rd
        self.model_hash = md.model_hash
        self.scores = md.scores
        self.chunks = []
        for ck in md.chunks:
            chunk = Chunk()
            chunk.deserialize(ck)
            self.chunks.append(chunk)

    def to_json(self):
        json_chunks = []
        if self.chunks:
            for c in self.chunks:
                json_chunks.append(c.to_json())
        json_obj = {"owner": self.owner, "hash": self.model_hash, "chunks": json_chunks}
        return json_obj


class Block(object):
    def __init__(self, timestamp=0, time_diff=1.0, rd=0, id=0, miner="", beta_string=b'', stake=None, models=None, scores=None) -> None:
        self.timestamp = timestamp
        self.time_diff = time_diff
        self.rd = rd
        self.id = id
        self.miner = miner
        self.beta_string = beta_string
        self.models = models
        self.stake = stake
        self.scores = scores

    def serialize(self, bk):
        bk.timestamp = self.timestamp
        bk.time_diff = self.time_diff
        bk.rd = self.rd
        bk.id = self.id
        bk.miner = self.miner
        bk.beta_string = self.beta_string
        bk.stake.update(self.stake)
        for m in self.models:
            md = bk.models.add()
            m.serialize(md)
        bk.scores.extend(self.scores)

    def deserialize(self, bk):
        self.timestamp = bk.timestamp
        self.time_diff = bk.time_diff
        self.rd = bk.rd
        self.id = bk.id
        self.miner = bk.miner
        self.beta_string = bk.beta_string
        self.stake = bk.stake
        self.models = []
        for md in bk.models:
            model = Model()
            model.deserialize(md)
            self.models.append(model)
        self.scores = []
        for s in bk.scores:
            self.scores.append(s)

    def to_json(self):
        json_models = []
        if self.models:
            for m in self.models:
                json_models.append(m.to_json())
        json_stake = {}
        for s in self.stake:
            json_stake[s] = self.stake[s]
        json_obj = {"timestamp": self.timestamp, "time difficulty": self.time_diff, "round": self.rd, "id": self.id, "miner": self.miner, "beta_string": self.beta_string.decode('latin-1'), "stake": json_stake, "models": json_models}
        return json_obj
