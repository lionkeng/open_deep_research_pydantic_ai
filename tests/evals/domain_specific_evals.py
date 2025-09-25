"""Domain-Specific Evaluators for ClarificationAgent.

This module provides specialized evaluators for different domains (technical, scientific,
business, creative) with domain-specific expertise and evaluation criteria.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.clarification import ClarifyWithUser


class Domain(Enum):
    """Supported evaluation domains."""

    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    BUSINESS = "business"
    CREATIVE = "creative"
    EDUCATIONAL = "educational"
    MEDICAL = "medical"


class DomainContext(BaseModel):
    """Context information for domain-specific evaluation."""

    domain: Domain
    subdomain: str | None = None
    technical_level: str | None = None  # beginner, intermediate, advanced
    industry: str | None = None
    specific_requirements: list[str] = Field(default_factory=list)


class DomainEvaluationResult(BaseModel):
    """Result from domain-specific evaluation."""

    domain: Domain
    overall_score: float = Field(ge=0, le=1)
    domain_relevance: float = Field(ge=0, le=1)
    expertise_accuracy: float = Field(ge=0, le=1)
    terminology_appropriateness: float = Field(ge=0, le=1)
    context_understanding: float = Field(ge=0, le=1)
    detailed_feedback: dict[str, Any] = Field(default_factory=dict)
    domain_specific_metrics: dict[str, float] = Field(default_factory=dict)


class BaseDomainEvaluator(ABC):
    """Base class for domain-specific evaluators."""

    def __init__(self, domain: Domain, model: str = "openai:gpt-5-mini"):
        self.domain = domain
        self.model = model
        self.domain_keywords = self._get_domain_keywords()
        self.domain_concepts = self._get_domain_concepts()
        self.quality_indicators = self._get_quality_indicators()

        # Create domain-specific judge agent
        self.judge_agent = Agent(model=model, system_prompt=self._create_domain_system_prompt())

    @abstractmethod
    def _get_domain_keywords(self) -> set[str]:
        """Get domain-specific keywords for analysis."""
        pass

    @abstractmethod
    def _get_domain_concepts(self) -> dict[str, list[str]]:
        """Get domain concepts and their related terms."""
        pass

    @abstractmethod
    def _get_quality_indicators(self) -> dict[str, list[str]]:
        """Get indicators of high-quality domain-specific clarifications."""
        pass

    @abstractmethod
    def _create_domain_system_prompt(self) -> str:
        """Create domain-specific system prompt for LLM judge."""
        pass

    @abstractmethod
    def domain_evaluate(
        self, output: ClarifyWithUser, context: DomainContext = None
    ) -> DomainEvaluationResult:
        """Evaluate clarification output for domain-specific quality."""
        pass

    @abstractmethod
    def _calculate_domain_metrics(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, float]:
        """Calculate domain-specific metrics."""
        pass

    @abstractmethod
    def _generate_detailed_feedback(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, Any]:
        """Generate detailed domain-specific feedback."""
        pass

    def evaluate(self, ctx) -> dict[str, Any]:
        """Wrapper to comply with Evaluator interface."""
        # Extract output from context
        output = ctx.output if hasattr(ctx, "output") else None
        if not output:
            return {"score": 0.0, "error": "No output available"}

        # Create domain context if available
        domain_context = None
        if hasattr(ctx, "metadata") and ctx.metadata:
            domain_context = DomainContext(
                query=ctx.inputs.query if hasattr(ctx.inputs, "query") else str(ctx.inputs),
                domain=self.domain,
            )

        # Call the domain-specific evaluate
        result = self.domain_evaluate(output, domain_context)

        # Convert DomainEvaluationResult to dict
        return {
            "score": result.overall_score,
            "domain": result.domain.value,
            "domain_relevance": result.domain_relevance,
            "expertise_accuracy": result.expertise_accuracy,
            "terminology_appropriateness": result.terminology_appropriateness,
            "context_understanding": result.context_understanding,
            "detailed_feedback": result.detailed_feedback,
        }

    def _analyze_domain_relevance(self, questions_text: str, context: DomainContext) -> float:
        """Analyze how relevant the questions are to the domain."""

        # Check for domain keywords
        keyword_matches = sum(1 for keyword in self.domain_keywords if keyword in questions_text)
        keyword_score = min(keyword_matches / 3, 1.0)  # Normalize to max 1.0

        # Check for domain concepts
        concept_matches = 0
        for concept, related_terms in self.domain_concepts.items():
            if concept in questions_text or any(term in questions_text for term in related_terms):
                concept_matches += 1

        concept_score = min(concept_matches / len(self.domain_concepts), 1.0)

        return (keyword_score + concept_score) / 2

    def _analyze_terminology(self, questions_text: str, context: DomainContext) -> float:
        """Analyze appropriateness of terminology used."""

        # Check for quality indicators
        quality_score = 0
        total_indicators = 0

        for _category, indicators in self.quality_indicators.items():
            category_matches = sum(1 for indicator in indicators if indicator in questions_text)
            if indicators:
                quality_score += category_matches / len(indicators)
                total_indicators += 1

        return quality_score / total_indicators if total_indicators > 0 else 0.5

    def _analyze_context_understanding(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> float:
        """Analyze how well the output understands the context."""
        # Basic implementation that can be overridden by subclasses
        if not output.request or not output.request.questions:
            return 0.0

        # Check if questions address missing dimensions
        if hasattr(output, 'missing_dimensions') and output.missing_dimensions:
            dimension_score = min(len(output.missing_dimensions) / 4, 1.0)
        else:
            dimension_score = 0.5

        # Check if questions are contextually appropriate
        if hasattr(output, 'assessment_reasoning') and output.assessment_reasoning:
            reasoning_score = min(len(output.assessment_reasoning) / 100, 1.0)
        else:
            reasoning_score = 0.5

        return (dimension_score + reasoning_score) / 2

    @abstractmethod
    def _calculate_domain_metrics(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, float]:
        """Calculate domain-specific metrics."""
        pass

    @abstractmethod
    def _generate_detailed_feedback(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, Any]:
        """Generate detailed feedback for the domain."""
        pass


class TechnicalDomainEvaluator(BaseDomainEvaluator):
    """Evaluator for technical/programming domain queries."""

    def __init__(self, model: str = "openai:gpt-5-mini"):
        super().__init__(Domain.TECHNICAL, model)

    def _get_domain_keywords(self) -> set[str]:
        return {
            "algorithm",
            "code",
            "programming",
            "software",
            "development",
            "architecture",
            "database",
            "api",
            "framework",
            "library",
            "debugging",
            "optimization",
            "performance",
            "scalability",
            "security",
            "testing",
            "deployment",
        }

    def _get_domain_concepts(self) -> dict[str, list[str]]:
        return {
            "programming_languages": ["python", "javascript", "java", "c++", "rust", "go"],
            "development_methodologies": ["agile", "devops", "ci/cd", "tdd", "microservices"],
            "system_design": ["load balancing", "caching", "distributed systems", "databases"],
            "security": ["authentication", "encryption", "vulnerabilities", "best practices"],
        }

    def _get_quality_indicators(self) -> dict[str, list[str]]:
        return {
            "specificity": ["specific", "particular", "exact", "precise"],
            "technical_depth": ["implementation", "architecture", "design", "technical"],
            "practical_focus": ["use case", "requirements", "constraints", "environment"],
        }

    def _create_domain_system_prompt(self) -> str:
        return """You are a technical expert evaluating clarification questions for
programming and software development queries.

        Evaluate the technical accuracy, appropriate use of terminology, and relevance to
software engineering practices.
        Focus on whether the questions would help identify technical requirements,
constraints, and implementation details.

        Return JSON with scores (0-10) and reasoning."""

    def _calculate_domain_metrics(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, float]:
        questions_text = " ".join(q.question for q in output.request.questions).lower()

        return {
            "technical_specificity": self._score_technical_specificity(questions_text),
            "implementation_focus": self._score_implementation_focus(questions_text),
            "architecture_consideration": self._score_architecture_consideration(questions_text),
            "tool_technology_relevance": self._score_tool_relevance(questions_text),
        }

    def _score_technical_specificity(self, text: str) -> float:
        specific_terms = [
            "version",
            "framework",
            "library",
            "platform",
            "environment",
            "implementation",
        ]
        matches = sum(1 for term in specific_terms if term in text)
        return min(matches / 3, 1.0)

    def _score_implementation_focus(self, text: str) -> float:
        impl_terms = ["how", "implement", "build", "create", "develop", "code"]
        matches = sum(1 for term in impl_terms if term in text)
        return min(matches / 2, 1.0)

    def _score_architecture_consideration(self, text: str) -> float:
        arch_terms = ["architecture", "design", "structure", "pattern", "scalability"]
        matches = sum(1 for term in arch_terms if term in text)
        return min(matches / 2, 1.0)

    def _score_tool_relevance(self, text: str) -> float:
        tools = ["framework", "library", "tool", "service", "platform", "database"]
        matches = sum(1 for tool in tools if tool in text)
        return min(matches / 2, 1.0)

    def _generate_detailed_feedback(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, Any]:
        questions = [q.question for q in output.request.questions]

        feedback = {"strengths": [], "weaknesses": [], "suggestions": [], "technical_coverage": {}}

        questions_text = " ".join(questions).lower()

        # Analyze strengths
        if "specific" in questions_text or "particular" in questions_text:
            feedback["strengths"].append("Questions ask for specific details")

        if any(tech in questions_text for tech in ["technology", "framework", "tool"]):
            feedback["strengths"].append("Questions address technology choices")

        # Analyze weaknesses
        if not any(impl in questions_text for impl in ["implement", "build", "develop"]):
            feedback["weaknesses"].append("Questions don't address implementation approach")

        if not any(req in questions_text for req in ["requirement", "constraint", "limitation"]):
            feedback["weaknesses"].append("Questions don't explore technical requirements")

        # Generate suggestions
        if "performance" not in questions_text:
            feedback["suggestions"].append("Consider asking about performance requirements")

        if "scale" not in questions_text and "scalability" not in questions_text:
            feedback["suggestions"].append("Consider asking about scale and scalability needs")

        return feedback

    def domain_evaluate(
        self, output: ClarifyWithUser, context: DomainContext = None
    ) -> DomainEvaluationResult:
        """Technical domain-specific evaluation."""

        if not output.needs_clarification or not output.request:
            return DomainEvaluationResult(
                domain=self.domain,
                overall_score=0.0,
                domain_relevance=0.0,
                expertise_accuracy=0.0,
                terminology_appropriateness=0.0,
                context_understanding=0.0,
                detailed_feedback={"applicable": False},
            )

        questions = [q.question for q in output.request.questions]
        questions_text = " ".join(questions).lower()

        # Analyze domain relevance
        domain_relevance = self._analyze_domain_relevance(questions_text, context)

        # Analyze terminology appropriateness
        terminology_score = self._analyze_terminology(questions_text, context)

        # Analyze context understanding
        context_score = self._analyze_context_understanding(output, context)

        # Get domain-specific metrics
        domain_metrics = self._calculate_domain_metrics(output, context)

        # Calculate overall score
        overall_score = (
            domain_relevance * 0.3
            + terminology_score * 0.25
            + context_score * 0.25
            + (sum(domain_metrics.values()) / len(domain_metrics) if domain_metrics else 0.5) * 0.2
        )

        return DomainEvaluationResult(
            domain=self.domain,
            overall_score=overall_score,
            domain_relevance=domain_relevance,
            expertise_accuracy=0.8,
            terminology_appropriateness=terminology_score,
            context_understanding=context_score,
            detailed_feedback=self._generate_detailed_feedback(output, context),
            domain_specific_metrics=domain_metrics,
        )


class ScientificDomainEvaluator(BaseDomainEvaluator):
    """Evaluator for scientific research domain queries."""

    def __init__(self, model: str = "openai:gpt-5-mini"):
        super().__init__(Domain.SCIENTIFIC, model)

    def _get_domain_keywords(self) -> set[str]:
        return {
            "research",
            "study",
            "analysis",
            "experiment",
            "hypothesis",
            "methodology",
            "data",
            "statistical",
            "peer-reviewed",
            "publication",
            "evidence",
            "theory",
            "scientific",
            "academic",
            "empirical",
            "quantitative",
            "qualitative",
        }

    def _get_domain_concepts(self) -> dict[str, list[str]]:
        return {
            "research_methods": ["experiment", "survey", "case study", "meta-analysis"],
            "data_analysis": ["statistics", "correlation", "regression", "significance"],
            "publication": ["peer-review", "journal", "citation", "publication"],
            "disciplines": ["biology", "chemistry", "physics", "psychology", "medicine"],
        }

    def _get_quality_indicators(self) -> dict[str, list[str]]:
        return {
            "methodological_rigor": ["methodology", "protocol", "controlled", "systematic"],
            "evidence_quality": ["peer-reviewed", "credible", "authoritative", "recent"],
            "scope_precision": ["specific field", "particular aspect", "focused area"],
        }

    def _create_domain_system_prompt(self) -> str:
        return """You are a scientific research expert evaluating clarification questions for research queries.

        Evaluate the scientific rigor, appropriate methodology considerations, and research relevance.
        Focus on whether the questions would help identify research scope, methodology, and evidence requirements.

        Return JSON with scores (0-10) and reasoning."""

    def _calculate_domain_metrics(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, float]:
        questions_text = " ".join(q.question for q in output.request.questions).lower()

        return {
            "methodological_awareness": self._score_methodology_awareness(questions_text),
            "research_scope_clarity": self._score_research_scope(questions_text),
            "evidence_requirements": self._score_evidence_requirements(questions_text),
            "academic_rigor": self._score_academic_rigor(questions_text),
        }

    def _score_methodology_awareness(self, text: str) -> float:
        method_terms = ["methodology", "method", "approach", "protocol", "systematic"]
        matches = sum(1 for term in method_terms if term in text)
        return min(matches / 2, 1.0)

    def _score_research_scope(self, text: str) -> float:
        scope_terms = ["field", "area", "discipline", "domain", "specific", "particular"]
        matches = sum(1 for term in scope_terms if term in text)
        return min(matches / 3, 1.0)

    def _score_evidence_requirements(self, text: str) -> float:
        evidence_terms = ["evidence", "source", "credible", "peer-reviewed", "authoritative"]
        matches = sum(1 for term in evidence_terms if term in text)
        return min(matches / 2, 1.0)

    def _score_academic_rigor(self, text: str) -> float:
        rigor_terms = ["academic", "scholarly", "research", "study", "analysis"]
        matches = sum(1 for term in rigor_terms if term in text)
        return min(matches / 2, 1.0)

    def _generate_detailed_feedback(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, Any]:
        questions_text = " ".join(q.question for q in output.request.questions).lower()

        feedback = {
            "methodological_coverage": "adequate"
            if "method" in questions_text
            else "needs_improvement",
            "scope_definition": "clear"
            if any(term in questions_text for term in ["specific", "field", "area"])
            else "vague",
            "evidence_standards": "addressed"
            if any(term in questions_text for term in ["source", "credible", "quality"])
            else "missing",
            "research_level": self._assess_research_level(questions_text),
            "suggestions": [],
        }

        if "hypothesis" not in questions_text:
            feedback["suggestions"].append(
                "Consider asking about research hypothesis or objectives"
            )

        if "time" not in questions_text and "recent" not in questions_text:
            feedback["suggestions"].append(
                "Consider asking about time frame or currency of research"
            )

        return feedback

    def _assess_research_level(self, text: str) -> str:
        if any(term in text for term in ["undergraduate", "basic", "introduction"]):
            return "undergraduate"
        elif any(term in text for term in ["graduate", "advanced", "specialized"]):
            return "graduate"
        elif any(term in text for term in ["cutting-edge", "latest", "novel"]):
            return "research"
        else:
            return "general"


class BusinessDomainEvaluator(BaseDomainEvaluator):
    """Evaluator for business and commercial domain queries."""

    def __init__(self, model: str = "openai:gpt-5-mini"):
        super().__init__(Domain.BUSINESS, model)

    def _get_domain_keywords(self) -> set[str]:
        return {
            "business",
            "company",
            "market",
            "revenue",
            "profit",
            "strategy",
            "competitive",
            "customer",
            "product",
            "service",
            "marketing",
            "sales",
            "operations",
            "finance",
            "management",
            "leadership",
            "organization",
            "industry",
            "sector",
            "roi",
        }

    def _get_domain_concepts(self) -> dict[str, list[str]]:
        return {
            "business_strategy": ["competitive advantage", "market positioning", "growth strategy"],
            "financial_metrics": ["roi", "profit margin", "cash flow", "revenue"],
            "market_analysis": ["market size", "target audience", "competition"],
            "operations": ["efficiency", "process optimization", "supply chain"],
        }

    def _get_quality_indicators(self) -> dict[str, list[str]]:
        return {
            "business_focus": ["roi", "value", "impact", "business case"],
            "stakeholder_awareness": ["stakeholder", "customer", "user", "client"],
            "practical_implementation": ["implementation", "execution", "practical", "actionable"],
        }

    def _create_domain_system_prompt(self) -> str:
        return """You are a business expert evaluating clarification questions for commercial and business queries.

        Evaluate the business relevance, stakeholder consideration, and practical applicability.
        Focus on whether the questions would help identify business objectives, constraints, and success metrics.

        Return JSON with scores (0-10) and reasoning."""

    def _calculate_domain_metrics(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, float]:
        questions_text = " ".join(q.question for q in output.request.questions).lower()

        return {
            "business_value_focus": self._score_business_value(questions_text),
            "stakeholder_consideration": self._score_stakeholder_awareness(questions_text),
            "market_context": self._score_market_context(questions_text),
            "implementation_practicality": self._score_implementation_practicality(questions_text),
        }

    def _score_business_value(self, text: str) -> float:
        value_terms = ["roi", "value", "benefit", "cost", "profit", "revenue", "impact"]
        matches = sum(1 for term in value_terms if term in text)
        return min(matches / 3, 1.0)

    def _score_stakeholder_awareness(self, text: str) -> float:
        stakeholder_terms = ["customer", "client", "user", "stakeholder", "team", "management"]
        matches = sum(1 for term in stakeholder_terms if term in text)
        return min(matches / 2, 1.0)

    def _score_market_context(self, text: str) -> float:
        market_terms = ["market", "industry", "competitive", "sector", "business environment"]
        matches = sum(1 for term in market_terms if term in text)
        return min(matches / 2, 1.0)

    def _score_implementation_practicality(self, text: str) -> float:
        practical_terms = ["practical", "feasible", "implementation", "execution", "actionable"]
        matches = sum(1 for term in practical_terms if term in text)
        return min(matches / 2, 1.0)

    def _generate_detailed_feedback(
        self, output: ClarifyWithUser, context: DomainContext
    ) -> dict[str, Any]:
        questions_text = " ".join(q.question for q in output.request.questions).lower()

        return {
            "business_alignment": "strong" if "business" in questions_text else "weak",
            "stakeholder_focus": "present"
            if any(term in questions_text for term in ["customer", "user", "client"])
            else "missing",
            "value_proposition": "addressed"
            if any(term in questions_text for term in ["value", "benefit", "roi"])
            else "unclear",
            "market_awareness": "demonstrated"
            if "market" in questions_text or "competitive" in questions_text
            else "lacking",
            "actionability": "high"
            if any(term in questions_text for term in ["how", "implement", "execute"])
            else "low",
        }


class DomainDetector:
    """Automatically detect the domain of a query for appropriate evaluator selection."""

    DOMAIN_INDICATORS = {
        Domain.TECHNICAL: [
            "code",
            "programming",
            "software",
            "algorithm",
            "database",
            "api",
            "framework",
            "debug",
            "implementation",
            "architecture",
            "development",
            "system",
        ],
        Domain.SCIENTIFIC: [
            "research",
            "study",
            "experiment",
            "hypothesis",
            "scientific",
            "analysis",
            "peer-reviewed",
            "methodology",
            "empirical",
            "statistical",
        ],
        Domain.BUSINESS: [
            "business",
            "market",
            "revenue",
            "profit",
            "customer",
            "strategy",
            "company",
            "sales",
            "marketing",
            "roi",
            "competitive",
            "industry",
        ],
        Domain.CREATIVE: [
            "creative",
            "design",
            "art",
            "artistic",
            "aesthetic",
            "creative",
            "innovative",
            "original",
            "inspiration",
            "concept",
            "visual",
        ],
        Domain.EDUCATIONAL: [
            "learn",
            "teaching",
            "education",
            "student",
            "course",
            "curriculum",
            "academic",
            "school",
            "university",
            "educational",
        ],
        Domain.MEDICAL: [
            "medical",
            "health",
            "disease",
            "treatment",
            "clinical",
            "patient",
            "diagnosis",
            "therapy",
            "healthcare",
            "medicine",
        ],
    }

    @classmethod
    def detect_domain(cls, query: str) -> Domain:
        """Detect the primary domain of a query."""
        query_lower = query.lower()

        domain_scores = {}
        for domain, indicators in cls.DOMAIN_INDICATORS.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            domain_scores[domain] = score

        # Return domain with highest score, or TECHNICAL as default
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return Domain.TECHNICAL

    @classmethod
    def get_evaluator_for_domain(
        cls, domain: Domain, model: str = "openai:gpt-5-mini"
    ) -> BaseDomainEvaluator:
        """Get appropriate evaluator for a domain."""
        evaluator_map = {
            Domain.TECHNICAL: TechnicalDomainEvaluator,
            Domain.SCIENTIFIC: ScientificDomainEvaluator,
            Domain.BUSINESS: BusinessDomainEvaluator,
            # Add more evaluators as needed
        }

        evaluator_class = evaluator_map.get(domain, TechnicalDomainEvaluator)
        return evaluator_class(model=model)

    @classmethod
    def evaluate_with_auto_detection(
        cls, query: str, output: ClarifyWithUser, context: DomainContext = None
    ) -> DomainEvaluationResult:
        """Automatically detect domain and evaluate."""
        detected_domain = cls.detect_domain(query)

        if context is None:
            context = DomainContext(domain=detected_domain)

        evaluator = cls.get_evaluator_for_domain(detected_domain)
        return evaluator.domain_evaluate(output, context)


# Example usage and testing
async def demo_domain_specific_evaluation():
    """Demonstrate domain-specific evaluation capabilities."""

    from models.clarification import (
        ClarificationChoice,
        ClarificationQuestion,
        ClarificationRequest,
    )

    # Technical query example
    technical_output = ClarifyWithUser(
        needs_clarification=True,
        request=ClarificationRequest(
            questions=[
                ClarificationQuestion(
                    question="What programming language are you planning to use for this implementation?",
                    question_type="choice",
                    choices=[
                        ClarificationChoice(id="py", label="Python"),
                        ClarificationChoice(id="js", label="JavaScript"),
                        ClarificationChoice(id="java", label="Java"),
                        ClarificationChoice(id="cpp", label="C++"),
                    ],
                    is_required=True,
                ),
                ClarificationQuestion(
                    question="What are your performance and scalability requirements?",
                    question_type="text",
                    is_required=True,
                ),
            ]
        ),
        missing_dimensions=["technical_requirements", "implementation_approach"],
        assessment_reasoning="The query lacks specific technical details and requirements.",
    )

    # Evaluate with technical domain evaluator
    technical_evaluator = TechnicalDomainEvaluator()
    tech_context = DomainContext(
        domain=Domain.TECHNICAL,
        technical_level="intermediate",
        specific_requirements=["performance", "scalability"],
    )

    result = technical_evaluator.domain_evaluate(technical_output, tech_context)

    print("=== Technical Domain Evaluation ===")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Domain Relevance: {result.domain_relevance:.3f}")
    print(f"Terminology Appropriateness: {result.terminology_appropriateness:.3f}")
    print(f"Context Understanding: {result.context_understanding:.3f}")
    print("\nDomain-Specific Metrics:")
    for metric, score in result.domain_specific_metrics.items():
        print(f"  {metric}: {score:.3f}")

    # Auto-detection example
    query = "How do I optimize my machine learning model for production deployment?"
    detected_domain = DomainDetector.detect_domain(query)
    print(f"\nAuto-detected domain for query: {detected_domain.value}")

    auto_result = DomainDetector.evaluate_with_auto_detection(query, technical_output)
    print(f"Auto-evaluation score: {auto_result.overall_score:.3f}")


class DomainEvaluationOrchestrator:
    """Orchestrator for domain-specific evaluations."""

    def __init__(self):
        self.detector = DomainDetector()
        self.evaluators = {}

    def evaluate(self, output, query: str = None) -> DomainEvaluationResult:
        """Evaluate output with appropriate domain evaluator."""
        # Detect domain from query
        if query:
            domain = DomainDetector.detect_domain(query)
        else:
            domain = Domain.TECHNICAL

        # Get appropriate evaluator
        evaluator = DomainDetector.get_evaluator_for_domain(domain)

        # Create context
        context = DomainContext(query=query, domain=domain) if query else None

        # Evaluate
        return evaluator.domain_evaluate(output, context)


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_domain_specific_evaluation())
