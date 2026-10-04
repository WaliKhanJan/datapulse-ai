import os, re, tempfile
from functools import lru_cache
import duckdb, lancedb, pandas as pd


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _docling():
    from docling.document_converter import DocumentConverter
    return DocumentConverter()


def _clean(name):
    n = re.sub(r"\W+", "_", str(name)).strip("_").lower()
    return "t_" + n if n[:1].isdigit() else n


def _chunks(text, size=800):
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > size and cur:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return out


class Store:
    """Holds DuckDB tables (structured) and a LanceDB index (PDF text)."""

    def __init__(self):
        self.con = duckdb.connect()
        self.tables = {}
        self.chunks = []
        self._lance = lancedb.connect(tempfile.mkdtemp())
        self._tbl = None

    def add_table(self, name, df):
        name = _clean(name)
        df, cols, seen = df.copy(), [], {}
        for i, c in enumerate(df.columns):
            c = _clean(c) or f"col{i}"
            seen[c] = seen.get(c, 0) + 1
            cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
        df.columns = cols
        self.con.register(name, df)
        self.tables[name] = df
        return name

    def add_file(self, path):
        stem, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lower()
        if ext == ".csv":
            self.add_table(stem, pd.read_csv(path))
        elif ext == ".parquet":
            self.add_table(stem, pd.read_parquet(path))
        elif ext == ".pdf":
            self._add_pdf(path, stem)

    def _add_pdf(self, path, stem):
        doc = _docling().convert(path).document
        for i, t in enumerate(doc.tables, 1):
            try:
                df = t.export_to_dataframe(doc=doc)
            except TypeError:
                df = t.export_to_dataframe()
            for c in df.columns:  # "1,234" / "$5" / "12%" -> numbers
                try:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[,$%\s]", "", regex=True))
                except (ValueError, TypeError):
                    pass
            if len(df):
                self.add_table(f"{stem}_table{i}", df)
        src = os.path.basename(path)
        self.chunks += [{"text": c, "source": src} for c in _chunks(doc.export_to_markdown())]
        self._reindex()

    def _reindex(self):
        vecs = _embedder().encode([c["text"] for c in self.chunks]).tolist()
        rows = [{**c, "vector": v} for c, v in zip(self.chunks, vecs)]
        self._tbl = self._lance.create_table("chunks", data=rows, mode="overwrite")

    def search(self, q, k=4):
        if self._tbl is None:
            return []
        v = _embedder().encode([q])[0].tolist()
        return [{"text": r["text"], "source": r["source"]} for r in self._tbl.search(v).limit(k).to_list()]

    def schema(self):
        out = []
        for n, df in self.tables.items():
            cols = ", ".join(f"{r[0]} {r[1]}" for r in self.con.execute(f'DESCRIBE "{n}"').fetchall())
            out.append(f"TABLE {n}({cols})\nSample rows:\n{df.head(3).to_csv(index=False)}")
        return "\n".join(out)
