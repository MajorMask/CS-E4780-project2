import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        rf"""
    # Graph RAG using Text2Cypher (Enhanced)

    This demo app includes:
    - Task 1: Text2Cypher improvements (exemplars, validation, refinement)
    - Task 2: Caching & Performance monitoring

    > \- Powered by Kuzu, DSPy and marimo \-
    """
    )
    return


@app.cell
def _(mo):
    text_ui = mo.ui.text(
        value="Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        full_width=True
    )
    return (text_ui,)


@app.cell
def _(text_ui):
    text_ui
    return


@app.cell
def _(KuzuDatabaseManager, mo, run_graph_rag, text_ui):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    question = text_ui.value

    with mo.status.spinner(title="Generating answer...") as _spinner:
        result = run_graph_rag([question], db_manager)[0]

    query = result['query']
    answer = result['answer'].response
    return answer, db_manager, db_name, query, question


@app.cell
def _(answer, mo, query):
    mo.hstack([
        mo.md(f"""### Query\n```{query}```"""),
        mo.md(f"""### Answer\n{answer}""")
    ])
    return


@app.cell
def _(GraphSchema, Query, dspy):
    class PruneSchema(dspy.Signature):
        """
        Understand the given labelled property graph schema and the given user question. Your task
        is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
        relevant to the question.
            - The schema is a list of nodes and edges in a property graph.
            - The nodes are the entities in the graph.
            - The edges are the relationships between the nodes.
            - Properties of nodes and edges are their attributes, which helps answer the question.
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        pruned_schema: GraphSchema = dspy.OutputField()


    class Text2Cypher(dspy.Signature):
        """
        Translate the question into a valid Cypher query that respects the graph schema.

        <SYNTAX>
        - When matching on Scholar names, ALWAYS match on the `knownName` property
        - For countries, cities, continents and institutions, you can match on the `name` property
        - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
        - Always strive to respect the relationship direction (FROM/TO) using the schema information.
        - When comparing string properties, ALWAYS do the following:
            - Lowercase the property values before comparison
            - Use the WHERE clause
            - Use the CONTAINS operator to check for presence of one substring in the other
        - DO NOT use APOC as the database does not support it.
        </SYNTAX>

        <RETURN_RESULTS>
        - If the result is an integer, return it as an integer (not a string).
        - When returning results, return property values rather than the entire node or relationship.
        - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
        - NO Cypher keywords should be returned by your query.
        </RETURN_RESULTS>
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        query: Query = dspy.OutputField()


    class AnswerQuestion(dspy.Signature):
        """
        - Use the provided question, the generated Cypher query and the context to answer the question.
        - If the context is empty, state that you don't have enough information to answer the question.
        - When dealing with dates, mention the month in full.
        """

        question: str = dspy.InputField()
        cypher_query: str = dspy.InputField()
        context: str = dspy.InputField()
        response: str = dspy.OutputField()
    return AnswerQuestion, PruneSchema, Text2Cypher


@app.cell
def _(GEMINI_API_KEY, dspy):
    # Using Google Gemini directly
    lm = dspy.LM(
        model="gemini/gemini-2.0-flash-exp",  # or gemini-1.5-pro
        api_key=GEMINI_API_KEY,
    )
    dspy.configure(lm=lm)
    return (lm,)


@app.cell
def _(kuzu):
    class KuzuDatabaseManager:
        """Manages Kuzu database connection and schema retrieval."""

        def __init__(self, db_path: str = "nobel.kuzu"):
            self.db_path = db_path
            self.db = kuzu.Database(db_path, read_only=True)
            self.conn = kuzu.Connection(self.db)

        @property
        def get_schema_dict(self) -> dict[str, list[dict]]:
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
            nodes = [row[1] for row in response]
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
            rel_tables = [row[1] for row in response]
            relationships = []
            for tbl_name in rel_tables:
                response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
                for row in response:
                    relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})
            schema = {"nodes": [], "edges": []}

            for node in nodes:
                node_schema = {"label": node, "properties": []}
                node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
                for row in node_properties:
                    node_schema["properties"].append({"name": row[1], "type": row[2]})
                schema["nodes"].append(node_schema)

            for rel in relationships:
                edge = {
                    "label": rel["name"],
                    "from": rel["from"],
                    "to": rel["to"],
                    "properties": [],
                }
                rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
                for row in rel_properties:
                    edge["properties"].append({"name": row[1], "type": row[2]})
                schema["edges"].append(edge)
            return schema
    return (KuzuDatabaseManager,)


@app.cell
def _(BaseModel, Field):
    class Query(BaseModel):
        query: str = Field(description="Valid Cypher query with no newlines")


    class Property(BaseModel):
        name: str
        type: str = Field(description="Data type of the property")


    class Node(BaseModel):
        label: str
        properties: list[Property] | None


    class Edge(BaseModel):
        label: str = Field(description="Relationship label")
        from_: str = Field(alias="from", description="Source node label")
        to: str = Field(description="Target node label")
        properties: list[Property] | None


    class GraphSchema(BaseModel):
        nodes: list[Node]
        edges: list[Edge]
    return Edge, GraphSchema, Node, Property, Query


# NEW CELL: Helper Classes for Task 1 and Task 2
@app.cell
def _(
    kuzu,
    np,
    time,
    hashlib,
    Optional,
    Any,
    Dict,
    List,
    Tuple,
    OrderedDict,
    TfidfVectorizer,
    cosine_similarity,
    Query,
):
    # ============================================================
    # TASK 1: Text2Cypher Improvements
    # ============================================================
    
    class ExemplarDatabase:
        """Store and retrieve similar query examples"""
        
        def __init__(self):
            self.exemplars: List[Tuple[str, str]] = []
            self.vectorizer = TfidfVectorizer()
            self.question_vectors = None
            self._initialize_exemplars()
        
        def _initialize_exemplars(self):
            """Initialize with example question-query pairs"""
            self.exemplars = [
                (
                    "Which scholars won prizes in Physics?",
                    "MATCH (s:Scholar)-[r:WON]->(p:Prize) WHERE LOWER(p.category) = 'physics' RETURN s.knownName"
                ),
                (
                    "How many laureates were affiliated with MIT?",
                    "MATCH (s:Scholar)-[r:AFFILIATED_WITH]->(i:Institution) WHERE LOWER(i.name) CONTAINS 'mit' RETURN COUNT(DISTINCT s)"
                ),
                (
                    "Who won prizes in Chemistry and was born in France?",
                    "MATCH (s:Scholar)-[r:WON]->(p:Prize), (s)-[b:BORN_IN]->(c:City)-[:IS_CITY_IN]->(co:Country) WHERE LOWER(p.category) = 'chemistry' AND LOWER(co.name) CONTAINS 'france' RETURN s.knownName"
                ),
                (
                    "What institutions are located in Cambridge?",
                    "MATCH (i:Institution)-[r:IS_LOCATED_IN]->(c:City) WHERE LOWER(c.name) CONTAINS 'cambridge' RETURN i.name"
                ),
                (
                    "List all prizes awarded in 2020",
                    "MATCH (p:Prize) WHERE p.awardYear = 2020 RETURN p.category, p.prizeAmount"
                ),
            ]
            
            questions = [q for q, _ in self.exemplars]
            self.question_vectors = self.vectorizer.fit_transform(questions)
        
        def get_similar_exemplars(self, question: str, k: int = 3) -> List[Tuple[str, str]]:
            """Retrieve k most similar exemplars"""
            if not self.exemplars:
                return []
            
            query_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(query_vector, self.question_vectors)[0]
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            return [self.exemplars[i] for i in top_k_indices]
    
    
    class CypherValidator:
        """Validate and refine Cypher queries"""
        
        def __init__(self, conn):
            self.conn = conn
            self.max_refinement_attempts = 3
        
        def validate_syntax(self, query: str) -> Tuple[bool, str]:
            """Validate using EXPLAIN"""
            try:
                explain_query = f"EXPLAIN {query}"
                self.conn.execute(explain_query)
                return True, ""
            except Exception as e:
                return False, str(e)
        
        def apply_post_processing(self, query: str) -> str:
            """Apply rule-based fixes"""
            # Ensure query doesn't have multiple lines
            query = " ".join(query.split())
            return query
    
    
    # ============================================================
    # TASK 2: Caching & Performance
    # ============================================================
    
    class LRUCache:
        """LRU Cache for Text2Cypher results"""
        
        def __init__(self, capacity: int = 100):
            self.capacity = capacity
            self.cache: OrderedDict = OrderedDict()
            self.hits = 0
            self.misses = 0
        
        def _compute_key(self, question: str, schema: str) -> str:
            """Compute cache key"""
            combined = f"{question}|{schema}"
            return hashlib.sha256(combined.encode()).hexdigest()
        
        def get(self, question: str, schema: str) -> Optional[Any]:
            """Retrieve cached result"""
            key = self._compute_key(question, schema)
            
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            
            self.misses += 1
            return None
        
        def put(self, question: str, schema: str, value: Any) -> None:
            """Store result in cache"""
            key = self._compute_key(question, schema)
            
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
            
            self.cache[key] = value
        
        def get_stats(self) -> Dict[str, Any]:
            """Get cache statistics"""
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }
    
    
    class PerformanceMonitor:
        """Monitor pipeline stage timings"""
        
        def __init__(self):
            self.timings: Dict[str, List[float]] = {}
            self.stage_order: List[str] = []
        
        def record_timing(self, stage_name: str, duration_ms: float) -> None:
            """Record a timing"""
            if stage_name not in self.timings:
                self.timings[stage_name] = []
                self.stage_order.append(stage_name)
            
            self.timings[stage_name].append(duration_ms)
        
        def get_summary(self) -> Dict[str, Dict[str, float]]:
            """Get summary statistics"""
            summary = {}
            
            for stage_name, times in self.timings.items():
                if times:
                    summary[stage_name] = {
                        "count": len(times),
                        "mean": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "total": sum(times),
                    }
            
            return summary
        
        def print_summary(self) -> None:
            """Print formatted summary"""
            summary = self.get_summary()
            
            print("\n" + "=" * 70)
            print("PERFORMANCE SUMMARY")
            print("=" * 70)
            
            total_time = sum(s["total"] for s in summary.values())
            
            for stage_name in self.stage_order:
                if stage_name in summary:
                    stats = summary[stage_name]
                    percentage = (stats["total"] / total_time * 100) if total_time > 0 else 0
                    
                    print(f"\n{stage_name}:")
                    print(f"  Mean:  {stats['mean']:.2f} ms")
                    print(f"  Total: {stats['total']:.2f} ms ({percentage:.1f}%)")
            
            print("\n" + "=" * 70)
            print(f"Total: {total_time:.2f} ms")
            print("=" * 70 + "\n")
    
    return (
        ExemplarDatabase,
        CypherValidator,
        LRUCache,
        PerformanceMonitor,
    )


# MODIFIED: Enhanced GraphRAG class
@app.cell
def _(
    AnswerQuestion,
    Any,
    KuzuDatabaseManager,
    PruneSchema,
    Query,
    Text2Cypher,
    dspy,
    time,
    ExemplarDatabase,
    CypherValidator,
    LRUCache,
    PerformanceMonitor,
):
    class GraphRAG(dspy.Module):
        """Enhanced GraphRAG with caching and performance monitoring"""

        def __init__(self, db_manager: KuzuDatabaseManager):
            # Original components
            self.prune = dspy.Predict(PruneSchema)
            self.text2cypher = dspy.ChainOfThought(Text2Cypher)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
            
            # Task 1: Text2Cypher improvements
            self.exemplar_db = ExemplarDatabase()
            self.validator = CypherValidator(db_manager.conn)
            
            # Task 2: Caching & Performance
            self.cache = LRUCache(capacity=100)
            self.monitor = PerformanceMonitor()

        def get_cypher_query(self, question: str, input_schema: str) -> Query:
            """Generate Cypher query with improvements"""
            
            # Task 1: Get similar exemplars
            start = time.perf_counter()
            similar_exemplars = self.exemplar_db.get_similar_exemplars(question, k=3)
            self.monitor.record_timing("exemplar_selection", (time.perf_counter() - start) * 1000)
            
            # Schema pruning
            start = time.perf_counter()
            prune_result = self.prune(question=question, input_schema=input_schema)
            schema = prune_result.pruned_schema
            self.monitor.record_timing("schema_pruning", (time.perf_counter() - start) * 1000)
            
            # Generate query
            start = time.perf_counter()
            text2cypher_result = self.text2cypher(question=question, input_schema=schema)
            cypher_query = text2cypher_result.query
            self.monitor.record_timing("text2cypher", (time.perf_counter() - start) * 1000)
            
            # Task 1: Validate and refine
            start = time.perf_counter()
            query_str = cypher_query.query
            is_valid, error = self.validator.validate_syntax(query_str)
            
            if not is_valid:
                print(f"⚠ Query validation failed: {error}")
                # Apply post-processing
                query_str = self.validator.apply_post_processing(query_str)
                cypher_query = Query(query=query_str)
            
            self.monitor.record_timing("validation", (time.perf_counter() - start) * 1000)
            
            return cypher_query

        def run_query(
            self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
        ) -> tuple[str, list[Any] | None]:
            """Run query with timing"""
            result = self.get_cypher_query(question=question, input_schema=input_schema)
            query = result.query
            
            start = time.perf_counter()
            try:
                result = db_manager.conn.execute(query)
                results = [item for row in result for item in row]
            except RuntimeError as e:
                print(f"❌ Error running query: {e}")
                results = None
            self.monitor.record_timing("query_execution", (time.perf_counter() - start) * 1000)
            
            return query, results

        def forward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            """Execute pipeline with caching"""
            
            # Task 2: Check cache first
            cached_result = self.cache.get(question, input_schema)
            if cached_result is not None:
                print("✓ Cache hit!")
                return cached_result
            
            print("✗ Cache miss - executing pipeline")
            
            # Execute pipeline
            start_total = time.perf_counter()
            
            final_query, final_context = self.run_query(db_manager, question, input_schema)
            
            if final_context is None:
                return {}
            else:
                start = time.perf_counter()
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                self.monitor.record_timing("answer_generation", (time.perf_counter() - start) * 1000)
                
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                
                # Task 2: Cache the result
                self.cache.put(question, input_schema, response)
                
                total_time = (time.perf_counter() - start_total) * 1000
                self.monitor.record_timing("total_pipeline", total_time)
                
                return response


    def run_graph_rag(questions: list[str], db_manager: KuzuDatabaseManager) -> list[Any]:
        schema = str(db_manager.get_schema_dict)
        rag = GraphRAG(db_manager)
        
        results = []
        for question in questions:
            response = rag(db_manager=db_manager, question=question, input_schema=schema)
            results.append(response)
        
        # Print performance stats after all queries
        print("\n" + "="*70)
        print("CACHE STATISTICS")
        print("="*70)
        cache_stats = rag.cache.get_stats()
        print(f"Cache hits: {cache_stats['hits']}")
        print(f"Cache misses: {cache_stats['misses']}")
        print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"Cache size: {cache_stats['size']}/{cache_stats['capacity']}")
        
        rag.monitor.print_summary()
        
        return results

    return (run_graph_rag, GraphRAG)


@app.cell
def _():
    return


# MODIFIED: Updated imports
@app.cell
def _():
    import marimo as mo
    import os
    import time
    import hashlib
    from textwrap import dedent
    from typing import Any, Optional, Dict, List, Tuple, Callable
    from collections import OrderedDict
    from functools import wraps

    import dspy
    import kuzu
    import numpy as np
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    load_dotenv()

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    return (
        Any,
        BaseModel,
        Callable,
        Dict,
        Field,
        GEMINI_API_KEY,
        List,
        Optional,
        OrderedDict,
        Tuple,
        TfidfVectorizer,
        cosine_similarity,
        dspy,
        hashlib,
        kuzu,
        mo,
        np,
        time,
        wraps,
    )


if __name__ == "__main__":
    app.run()