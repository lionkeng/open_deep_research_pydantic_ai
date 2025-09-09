"""Data models for research plans and enhanced query transformation."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .search_query_models import SearchQueryBatch


class ResearchObjective(BaseModel):
    """Individual research objective with success criteria."""

    id: str = Field(description="Unique identifier for this objective")
    objective: str = Field(description="Clear statement of what to achieve")
    priority: Literal["PRIMARY", "SECONDARY", "TERTIARY"] = Field(
        default="SECONDARY", description="Priority level of this objective"
    )
    success_criteria: str = Field(description="How to measure if this objective is achieved")
    key_questions: list[str] = Field(
        default_factory=list, description="Key questions to answer for this objective"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of objectives this depends on"
    )

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, v: str) -> str:
        """Ensure objective is well-formed."""
        v = v.strip()
        if not v:
            raise ValueError("Objective cannot be empty")
        if len(v) < 10:
            raise ValueError("Objective must be at least 10 characters")
        return v


class ResearchMethodology(BaseModel):
    """Research methodology and approach."""

    approach: str = Field(description="Overall research approach")
    data_sources: list[str] = Field(description="Types of data sources to use")
    analysis_methods: list[str] = Field(description="Methods for analyzing gathered information")
    quality_criteria: list[str] = Field(
        default_factory=list, description="Criteria for evaluating information quality"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Known limitations of the approach"
    )


class ResearchPlan(BaseModel):
    """Comprehensive research plan with objectives and methodology."""

    objectives: list[ResearchObjective] = Field(
        ..., min_length=1, max_length=10, description="Research objectives"
    )
    methodology: ResearchMethodology = Field(description="Research methodology")
    expected_deliverables: list[str] = Field(description="Expected outputs from the research")
    scope_definition: str = Field(default="", description="Clear definition of research scope")
    constraints: list[str] = Field(
        default_factory=list, description="Known constraints or limitations"
    )
    success_metrics: list[str] = Field(
        default_factory=list, description="How to measure overall success"
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "ResearchPlan":
        """Validate the research plan structure."""
        # Ensure at least one primary objective
        primary_count = sum(1 for obj in self.objectives if obj.priority == "PRIMARY")
        if primary_count == 0:
            # Set the first objective as primary
            if self.objectives:
                self.objectives[0].priority = "PRIMARY"
        elif primary_count > 3:
            raise ValueError("Too many primary objectives (max 3)")

        # Validate unique objective IDs
        obj_ids = [obj.id for obj in self.objectives]
        if len(obj_ids) != len(set(obj_ids)):
            raise ValueError("Objective IDs must be unique")

        # Validate dependencies exist
        for obj in self.objectives:
            for dep_id in obj.dependencies:
                if dep_id not in obj_ids:
                    raise ValueError(f"Dependency {dep_id} not found in objectives")

        return self

    def get_primary_objectives(self) -> list[ResearchObjective]:
        """Get all primary objectives."""
        return [obj for obj in self.objectives if obj.priority == "PRIMARY"]

    def get_objectives_by_priority(
        self, priority: Literal["PRIMARY", "SECONDARY", "TERTIARY"]
    ) -> list[ResearchObjective]:
        """Get objectives by priority level."""
        return [obj for obj in self.objectives if obj.priority == priority]

    def get_dependency_order(self) -> list[ResearchObjective]:
        """Get objectives in dependency order."""
        # Simple topological sort
        visited = set()
        result = []

        def visit(obj_id: str):
            if obj_id in visited:
                return
            visited.add(obj_id)
            obj = next((o for o in self.objectives if o.id == obj_id), None)
            if obj:
                for dep in obj.dependencies:
                    visit(dep)
                result.append(obj)

        for obj in self.objectives:
            visit(obj.id)

        return result


class EnhancedTransformedQuery(BaseModel):
    """Enhanced query transformation output combining search queries and research plan."""

    # Original context
    original_query: str = Field(description="The original user query")
    clarification_context: dict[str, Any] = Field(
        default_factory=dict, description="Context from clarification phase"
    )

    # Core outputs
    search_queries: SearchQueryBatch = Field(description="Batch of search queries to execute")
    research_plan: ResearchPlan = Field(description="Comprehensive research plan")

    # Transformation metadata
    transformation_rationale: str = Field(
        default="", description="Explanation of transformation approach"
    )
    confidence_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in transformation quality"
    )

    # Tracking
    ambiguities_resolved: list[str] = Field(
        default_factory=list, description="Ambiguities resolved during transformation"
    )
    assumptions_made: list[str] = Field(
        default_factory=list, description="Assumptions made during transformation"
    )
    potential_gaps: list[str] = Field(
        default_factory=list, description="Potential information gaps identified"
    )

    @model_validator(mode="after")
    def validate_coherence(self) -> "EnhancedTransformedQuery":
        """Ensure queries and plan are coherent."""
        # Validate that queries reference valid objectives
        objective_ids = {obj.id for obj in self.research_plan.objectives}
        for query in self.search_queries.queries:
            if query.objective_id and query.objective_id not in objective_ids:
                raise ValueError(
                    f"Query {query.id} references non-existent objective {query.objective_id}"
                )

        # Ensure all objectives have at least one query
        objectives_with_queries = {
            q.objective_id for q in self.search_queries.queries if q.objective_id
        }
        for obj in self.research_plan.objectives:
            if obj.priority == "PRIMARY" and obj.id not in objectives_with_queries:
                # Primary objectives must have queries
                raise ValueError(f"Primary objective {obj.id} has no associated queries")

        # Adjust confidence based on assumptions and gaps
        if len(self.assumptions_made) > 5:
            self.confidence_score = min(0.7, self.confidence_score)
        if len(self.potential_gaps) > 3:
            self.confidence_score = min(0.8, self.confidence_score)

        return self

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of the execution plan."""
        return {
            "total_queries": len(self.search_queries.queries),
            "execution_strategy": self.search_queries.execution_strategy.value,
            "primary_objectives": len(self.research_plan.get_primary_objectives()),
            "total_objectives": len(self.research_plan.objectives),
            "confidence": self.confidence_score,
            "has_assumptions": len(self.assumptions_made) > 0,
            "has_gaps": len(self.potential_gaps) > 0,
        }
