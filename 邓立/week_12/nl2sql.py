import sqlite3
import pandas as pd
from typing import Union, List, Tuple
from openai import OpenAI


class DBParser:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.table_names = self._get_table_names()
        self.foreign_keys = self._get_foreign_keys()

    def _get_table_names(self) -> List[str]:
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [r[0] for r in self.cursor.fetchall()]

    def _get_foreign_keys(self) -> List[dict]:
        fks = []
        for table in self.table_names:
            self.cursor.execute(f"PRAGMA foreign_key_list('{table}')")
            for row in self.cursor.fetchall():
                fks.append({
                    "constrained_table": table,
                    "constrained_columns": [row[3]],
                    "referred_table": row[2],
                    "referred_columns": [row[4]],
                })
        return fks

    def get_schema_text(self) -> str:
        lines = []
        for table in self.table_names:
            self.cursor.execute(f"PRAGMA table_info('{table}')")
            cols = self.cursor.fetchall()
            col_strs = [f"    {c[1]} {c[2]}" + (" PRIMARY KEY" if c[5] else "") for c in cols]
            lines.append(f"Table: {table}")
            lines.extend(col_strs)
            lines.append("")
        if self.foreign_keys:
            lines.append("Foreign key relationships:")
            for fk in self.foreign_keys:
                lines.append(
                    f"  {fk['constrained_table']}.{fk['constrained_columns'][0]}"
                    f" -> {fk['referred_table']}.{fk['referred_columns'][0]}"
                )
        return "\n".join(lines)

    def get_sample_data(self, table_name: str, n: int = 3) -> str:
        query = f"SELECT * FROM \"{table_name}\" LIMIT {n}"
        df = pd.read_sql_query(query, self.conn)
        return df.to_markdown(index=False)

    def check_sql(self, sql: str) -> Tuple[bool, str]:
        try:
            self.conn.execute(sql)
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def execute_sql(self, sql: str) -> List[tuple]:
        self.cursor.execute(sql)
        return self.cursor.fetchall()


class SQLAgent:
    SYSTEM_PROMPT = """You are an expert SQL developer working with a SQLite database.
Your task is to convert a natural language question into a valid SQLite SQL query.

Rules:
- Output ONLY the SQL query, nothing else. No markdown code fences, no explanations.
- Use double quotes around table/column names that have special characters (if needed, but the schema uses simple names so usually not needed).
- Use single quotes for string literals.
- When asked "how many tables", query sqlite_master: SELECT count(*) FROM sqlite_master WHERE type='table'
- When asked about records/rows in a table, use SELECT count(*) FROM tablename.
- Ensure the SQL is syntactically correct SQLite SQL.

Database schema:
{schema}

Some sample data from key tables for reference:
{samples}"""

    def __init__(self, db_path: str, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4/"):
        self.parser = DBParser(db_path)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # Build the schema context
        self.schema_text = self.parser.get_schema_text()

        # Collect samples from a few key tables (exclude sqlite internal tables)
        sample_tables = [t for t in self.parser.table_names if not t.startswith("sqlite_")]
        samples_parts = []
        for t in sample_tables[:6]:
            samples_parts.append(f"--- {t} ---\n{self.parser.get_sample_data(t)}")
        self.samples_text = "\n\n".join(samples_parts)

    def ask(self, question: str, model: str = "deepseek-v4-flash", max_retries: int = 3) -> str:
        user_prompt = f"Question: {question}\n\nGenerate the SQL query:"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(
                schema=self.schema_text,
                samples=self.samples_text
            )},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    top_p=0.9,
                )
                sql = completion.choices[0].message.content.strip()
                sql = sql.strip("`").strip()
                if sql.lower().startswith("sql"):
                    sql = sql[3:].strip()
                if sql.startswith("```"):
                    lines = sql.split("\n")
                    sql = "\n".join(lines[1:-1]) if len(lines) >= 3 else sql

                # Check if SQL is valid
                valid, err = self.parser.check_sql(sql)
                if not valid:
                    messages.append({"role": "assistant", "content": sql})
                    messages.append({"role": "user", "content": f"That SQL had an error: {err}\nPlease fix it and output only the corrected SQL."})
                    continue

                # Execute SQL
                result = self.parser.execute_sql(sql)
                return self._format_result(question, sql, result)

            except Exception as e:
                if attempt < max_retries - 1:
                    messages.append({"role": "user", "content": f"API call failed: {e}\nPlease try again."})
                else:
                    return f"Error: Failed to generate SQL after {max_retries} attempts. Last error: {e}"

        return "Error: Could not generate a valid SQL query."

    def _format_result(self, question: str, sql: str, result: List[tuple]) -> str:
        if not result:
            return f"[SQL] {sql}\n[结果] 查询无结果"

        if len(result) == 1 and len(result[0]) == 1:
            return f"[SQL] {sql}\n[结果] {result[0][0]}"
        elif len(result) == 1:
            return f"[SQL] {sql}\n[结果] {', '.join(str(v) for v in result[0])}"
        else:
            lines = [f"[SQL] {sql}", "[结果]"]
            for row in result[:50]:
                lines.append(f"  {row}")
            if len(result) > 50:
                lines.append(f"  ... (共 {len(result)} 行)")
            return "\n".join(lines)


if __name__ == "__main__":
    import os

    db_path = os.path.join(os.path.dirname(__file__), "chinook.db")

    api_key = "sk-4e73218205ba4dbcb15f49ba7bde310e"
    base_url = "https://api.deepseek.com/v1"

    agent = SQLAgent(db_path=db_path, api_key=api_key, base_url=base_url)

    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "数据库中所有客户个数和员工个数分别是多少",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"提问: {q}")
        answer = agent.ask(q)
        print(answer)
