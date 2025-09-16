"""
Enhanced Query Transformation Evaluators for comprehensive behavioral coverage.

This module provides additional evaluators that cover previously untested aspects
of the Query Transformation agent's behavior, including assumption quality,
priority distribution, clarification integration, and more.
"""

import re
from collections import Counter

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from models.research_plan_models import TransformedQuery


class AssumptionQualityEvaluator(Evaluator):
    """
    Evaluates the quality and reasonableness of assumptions made during transformation.

    Metrics:
    - Assumption count appropriateness
    - Assumption explicitness and clarity
    - Risk assessment of assumptions
    - Coverage of identified gaps
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate assumption quality."""
        output: TransformedQuery = ctx.output

        scores = {
            "count_appropriateness": self._evaluate_assumption_count(output),
            "clarity": self._evaluate_assumption_clarity(output),
            "gap_coverage": self._evaluate_gap_coverage(output),
            "risk_assessment": self._evaluate_assumption_risk(output)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_assumption_count(self, output: TransformedQuery) -> float:
        """Evaluate if assumption count is appropriate."""
        assumption_count = len(output.assumptions_made)
        gap_count = len(output.potential_gaps)

        # Ideal: assumptions should address gaps but not be excessive
        if gap_count == 0:
            # No gaps identified
            if assumption_count == 0:
                return 1.0  # Perfect - no gaps, no assumptions
            elif assumption_count <= 2:
                return 0.8  # Minor assumptions without gaps is okay
            else:
                return 0.5  # Too many assumptions without gaps
        else:
            # Gaps exist
            ratio = assumption_count / gap_count
            if 0.5 <= ratio <= 1.5:
                return 1.0  # Good ratio of assumptions to gaps
            elif 0.3 <= ratio < 0.5 or 1.5 < ratio <= 2.0:
                return 0.7  # Acceptable but not ideal
            else:
                return 0.4  # Poor ratio

    def _evaluate_assumption_clarity(self, output: TransformedQuery) -> float:
        """Evaluate clarity and explicitness of assumptions."""
        if not output.assumptions_made:
            return 1.0  # No assumptions is fine if no gaps

        clarity_scores = []
        for assumption in output.assumptions_made:
            score = 0.0

            # Check for explicit statement structure
            if any(keyword in assumption.lower()
                   for keyword in ["assume", "assuming", "presumed", "expected"]):
                score += 0.3

            # Check for justification
            if any(word in assumption.lower()
                   for word in ["because", "since", "based on", "given"]):
                score += 0.3

            # Check for specificity (longer, detailed assumptions)
            if len(assumption.split()) >= 8:
                score += 0.4

            clarity_scores.append(score)

        return sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0

    def _evaluate_gap_coverage(self, output: TransformedQuery) -> float:
        """Evaluate how well assumptions cover identified gaps."""
        if not output.potential_gaps:
            return 1.0  # No gaps to cover

        if not output.assumptions_made:
            return 0.0  # Gaps exist but no assumptions made

        # Simple heuristic: check if key terms from gaps appear in assumptions
        gap_terms = set()
        for gap in output.potential_gaps:
            gap_terms.update(word.lower() for word in gap.split() if len(word) > 3)

        assumption_terms = set()
        for assumption in output.assumptions_made:
            assumption_terms.update(word.lower() for word in assumption.split() if len(word) > 3)

        if not gap_terms:
            return 1.0

        coverage = len(gap_terms.intersection(assumption_terms)) / len(gap_terms)
        return min(1.0, coverage * 1.2)  # Boost slightly as this is a rough metric

    def _evaluate_assumption_risk(self, output: TransformedQuery) -> float:
        """Evaluate risk level of assumptions made."""
        if not output.assumptions_made:
            return 1.0

        # Penalize based on confidence score and assumption count
        assumption_penalty = min(len(output.assumptions_made) * 0.1, 0.5)

        # Lower confidence should correlate with more assumptions
        if output.confidence_score <= 0.6 and len(output.assumptions_made) > 3:
            return 0.9  # Good - low confidence with many assumptions
        elif output.confidence_score >= 0.9 and len(output.assumptions_made) > 5:
            return 0.4  # Bad - high confidence despite many assumptions
        else:
            return max(0.5, 1.0 - assumption_penalty)


class PriorityDistributionEvaluator(Evaluator):
    """
    Evaluates the distribution and appropriateness of search query priorities.

    Metrics:
    - Priority distribution balance
    - Alignment with objective importance
    - Critical query prioritization
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate priority distribution."""
        output: TransformedQuery = ctx.output

        scores = {
            "distribution_balance": self._evaluate_distribution(output),
            "objective_alignment": self._evaluate_objective_alignment(output),
            "critical_prioritization": self._evaluate_critical_queries(output)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_distribution(self, output: TransformedQuery) -> float:
        """Evaluate priority distribution balance."""
        queries = output.search_queries.queries
        if not queries:
            return 0.0

        priority_counts = Counter(q.priority for q in queries)
        total = len(queries)

        # Ideal distribution: 40-50% high (1-2), 30-40% medium (3), 20-30% low (4-5)
        high_ratio = (priority_counts.get(1, 0) + priority_counts.get(2, 0)) / total
        medium_ratio = priority_counts.get(3, 0) / total
        low_ratio = (priority_counts.get(4, 0) + priority_counts.get(5, 0)) / total

        score = 0.0

        # Check high priority
        if 0.3 <= high_ratio <= 0.6:
            score += 0.4
        elif 0.2 <= high_ratio < 0.3 or 0.6 < high_ratio <= 0.7:
            score += 0.2

        # Check medium priority
        if 0.2 <= medium_ratio <= 0.5:
            score += 0.3
        elif 0.1 <= medium_ratio < 0.2 or 0.5 < medium_ratio <= 0.6:
            score += 0.15

        # Check low priority
        if 0.1 <= low_ratio <= 0.4:
            score += 0.3
        elif 0.05 <= low_ratio < 0.1 or 0.4 < low_ratio <= 0.5:
            score += 0.15

        return score

    def _evaluate_objective_alignment(self, output: TransformedQuery) -> float:
        """Evaluate if query priorities align with objective priorities."""
        primary_objectives = output.research_plan.get_primary_objectives()
        if not primary_objectives:
            return 0.5  # No primary objectives to align with

        primary_obj_ids = {obj.id for obj in primary_objectives}

        # High priority queries should be linked to primary objectives
        high_priority_queries = [q for q in output.search_queries.queries if q.priority <= 2]
        if not high_priority_queries:
            return 0.3  # No high priority queries is problematic

        aligned_count = sum(
            1 for q in high_priority_queries
            if q.objective_id and q.objective_id in primary_obj_ids
        )

        alignment_ratio = aligned_count / len(high_priority_queries)
        return alignment_ratio

    def _evaluate_critical_queries(self, output: TransformedQuery) -> float:
        """Evaluate if critical/core queries are properly prioritized."""
        # Check if queries containing core terms from original query have high priority
        original_terms = set(output.original_query.lower().split())

        critical_queries = []
        for query in output.search_queries.queries:
            query_terms = set(query.query.lower().split())
            overlap = len(original_terms.intersection(query_terms))
            if overlap >= len(original_terms) * 0.5:  # At least 50% overlap
                critical_queries.append(query)

        if not critical_queries:
            return 1.0  # No clearly critical queries identified

        # Check if critical queries have high priority (1 or 2)
        high_priority_critical = sum(1 for q in critical_queries if q.priority <= 2)
        return high_priority_critical / len(critical_queries)


class ClarificationIntegrationEvaluator(Evaluator):
    """
    Evaluates how well clarification responses are integrated into the transformation.

    Metrics:
    - Coverage of answered questions
    - Preservation of user intent
    - Handling of skipped/partial responses
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate clarification integration."""
        output: TransformedQuery = ctx.output

        # Check if there's clarification context
        if not output.clarification_context:
            return 1.0  # No clarification to integrate

        scores = {
            "response_coverage": self._evaluate_response_coverage(output),
            "intent_preservation": self._evaluate_intent_preservation(output),
            "ambiguity_resolution": self._evaluate_ambiguity_resolution(output)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_response_coverage(self, output: TransformedQuery) -> float:
        """Evaluate if clarification responses are reflected in transformation."""
        clarification = output.clarification_context

        # Extract user responses if available
        if "answers" not in clarification or not clarification["answers"]:
            return 1.0  # No answers to integrate

        answers = clarification["answers"]
        if isinstance(answers, str):
            # Parse answers text to extract key terms
            answer_terms = set()
            for line in answers.split('\n'):
                if 'A:' in line and '[SKIPPED]' not in line:
                    answer_text = line.split('A:', 1)[1].strip()
                    answer_terms.update(word.lower() for word in answer_text.split()
                                      if len(word) > 3)
        else:
            return 0.5  # Unknown format

        if not answer_terms:
            return 1.0  # No substantive answers

        # Check if answer terms appear in transformation
        transformation_text = " ".join([
            output.research_plan.methodology.approach if output.research_plan.methodology else "",
            " ".join(obj.objective for obj in output.research_plan.objectives),
            " ".join(q.query for q in output.search_queries.queries)
        ]).lower()

        covered = sum(1 for term in answer_terms if term in transformation_text)
        coverage = covered / len(answer_terms) if answer_terms else 0.0

        return min(1.0, coverage * 1.5)  # Boost as this is approximate

    def _evaluate_intent_preservation(self, output: TransformedQuery) -> float:
        """Evaluate if user's clarified intent is preserved."""
        # Check if ambiguities were resolved
        if output.ambiguities_resolved:
            # More resolved ambiguities indicates better integration
            resolution_score = min(1.0, len(output.ambiguities_resolved) * 0.2)
        else:
            resolution_score = 0.3

        # Check confidence score - should be higher with clarification
        if output.confidence_score >= 0.8:
            confidence_bonus = 0.3
        elif output.confidence_score >= 0.7:
            confidence_bonus = 0.2
        else:
            confidence_bonus = 0.0

        return min(1.0, resolution_score + confidence_bonus)

    def _evaluate_ambiguity_resolution(self, output: TransformedQuery) -> float:
        """Evaluate how well ambiguities were resolved."""
        clarification = output.clarification_context

        # Check if missing dimensions were addressed
        if "missing_dimensions" in clarification:
            if output.ambiguities_resolved:
                # Good - ambiguities were explicitly resolved
                return 0.9
            elif output.assumptions_made:
                # Okay - assumptions were made for missing dimensions
                return 0.6
            else:
                # Poor - no resolution or assumptions
                return 0.3

        return 0.7  # Default neutral score


class QueryDecompositionEvaluator(Evaluator):
    """
    Evaluates the quality of query decomposition into sub-components.

    Metrics:
    - Decomposition structure (hierarchy)
    - Component independence
    - Coverage completeness
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate query decomposition quality."""
        output: TransformedQuery = ctx.output

        scores = {
            "hierarchy_quality": self._evaluate_hierarchy(output),
            "component_independence": self._evaluate_independence(output),
            "coverage_completeness": self._evaluate_coverage(output)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_hierarchy(self, output: TransformedQuery) -> float:
        """Evaluate hierarchical structure of objectives."""
        objectives = output.research_plan.objectives
        if len(objectives) < 2:
            return 0.5  # Too few objectives to have hierarchy

        # Check for priority levels (PRIMARY, SECONDARY, TERTIARY)
        priority_counts = Counter(obj.priority for obj in objectives)

        has_primary = priority_counts.get("PRIMARY", 0) > 0
        has_secondary = priority_counts.get("SECONDARY", 0) > 0
        has_tertiary = priority_counts.get("TERTIARY", 0) > 0

        score = 0.0
        if has_primary:
            score += 0.4
        if has_secondary:
            score += 0.3
        if has_primary and has_secondary:
            score += 0.2  # Bonus for multi-level
        if has_tertiary and has_secondary and has_primary:
            score += 0.1  # Bonus for full hierarchy

        return score

    def _evaluate_independence(self, output: TransformedQuery) -> float:
        """Evaluate independence of decomposed components."""
        objectives = output.research_plan.objectives
        if len(objectives) < 2:
            return 1.0  # Single objective is inherently independent

        # Check for minimal overlap in objective text
        objective_texts = [obj.objective.lower() for obj in objectives]

        overlap_scores = []
        for i, text1 in enumerate(objective_texts):
            for _j, text2 in enumerate(objective_texts[i+1:], i+1):
                words1 = set(text1.split())
                words2 = set(text2.split())

                # Remove common words
                common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
                words1 = words1 - common_words
                words2 = words2 - common_words

                if words1 and words2:
                    overlap = len(words1.intersection(words2))
                    max_size = max(len(words1), len(words2))
                    overlap_ratio = overlap / max_size

                    # Lower overlap is better for independence
                    independence = 1.0 - overlap_ratio
                    overlap_scores.append(independence)

        return sum(overlap_scores) / len(overlap_scores) if overlap_scores else 1.0

    def _evaluate_coverage(self, output: TransformedQuery) -> float:
        """Evaluate if decomposition covers all aspects of original query."""
        original_terms = set(output.original_query.lower().split())

        # Collect all terms from objectives
        objective_terms = set()
        for obj in output.research_plan.objectives:
            objective_terms.update(obj.objective.lower().split())

        # Check coverage
        covered = original_terms.intersection(objective_terms)
        coverage = len(covered) / len(original_terms) if original_terms else 0.0

        # Also check if objectives have supporting questions
        objectives_with_questions = sum(
            1 for obj in output.research_plan.objectives if obj.key_questions
        )
        question_coverage = objectives_with_questions / len(output.research_plan.objectives)

        return (coverage + question_coverage) / 2


class SupportingQuestionsEvaluator(Evaluator):
    """
    Evaluates the quality of supporting questions in research objectives.

    Metrics:
    - Question relevance to objectives
    - Question specificity
    - Coverage of knowledge gaps
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate supporting questions quality."""
        output: TransformedQuery = ctx.output

        # Collect all supporting questions
        all_questions = []
        for obj in output.research_plan.objectives:
            all_questions.extend(obj.key_questions)

        if not all_questions:
            return 0.3  # No supporting questions is problematic

        scores = {
            "relevance": self._evaluate_relevance(output, all_questions),
            "specificity": self._evaluate_specificity(all_questions),
            "diversity": self._evaluate_diversity(all_questions)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_relevance(self, output: TransformedQuery, questions: list[str]) -> float:
        """Evaluate relevance of questions to objectives."""
        relevance_scores = []

        for obj in output.research_plan.objectives:
            if not obj.key_questions:
                continue

            obj_terms = set(obj.objective.lower().split())
            for question in obj.key_questions:
                q_terms = set(question.lower().split())
                overlap = len(obj_terms.intersection(q_terms))
                relevance = overlap / len(obj_terms) if obj_terms else 0.0
                relevance_scores.append(relevance)

        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    def _evaluate_specificity(self, questions: list[str]) -> float:
        """Evaluate specificity and clarity of questions."""
        specificity_scores = []

        question_words = {"what", "when", "where", "why", "how", "which", "who"}

        for question in questions:
            score = 0.0

            # Check for question words
            if any(word in question.lower().split() for word in question_words):
                score += 0.3

            # Check for specificity (longer questions tend to be more specific)
            word_count = len(question.split())
            if word_count >= 7:
                score += 0.4
            elif word_count >= 5:
                score += 0.2

            # Check for question mark
            if question.strip().endswith("?"):
                score += 0.3

            specificity_scores.append(score)

        return sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0.0

    def _evaluate_diversity(self, questions: list[str]) -> float:
        """Evaluate diversity of questions."""
        if len(questions) <= 1:
            return 1.0

        # Check for diverse question types
        question_starts = [q.split()[0].lower() if q.split() else "" for q in questions]
        unique_starts = len(set(question_starts))
        diversity = unique_starts / len(questions)

        return diversity


class SuccessCriteriaMeasurabilityEvaluator(Evaluator):
    """
    Evaluates the measurability of success criteria in research objectives.

    Metrics:
    - Presence of quantifiable metrics
    - Clarity of completion indicators
    - Achievability assessment
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate success criteria measurability."""
        output: TransformedQuery = ctx.output

        all_criteria = []
        for obj in output.research_plan.objectives:
            if obj.success_criteria:
                all_criteria.append(obj.success_criteria)

        # Also include plan-level success metrics
        all_criteria.extend(output.research_plan.success_metrics)

        if not all_criteria:
            return 0.2  # No success criteria is very problematic

        scores = {
            "quantifiability": self._evaluate_quantifiability(all_criteria),
            "clarity": self._evaluate_clarity(all_criteria),
            "achievability": self._evaluate_achievability(all_criteria)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_quantifiability(self, criteria: list[str]) -> float:
        """Evaluate if criteria contain quantifiable metrics."""
        quantifiable_count = 0

        # Patterns that indicate quantifiable metrics
        quantifiable_patterns = [
            r'\d+',  # Contains numbers
            r'percentage|percent|%',  # Percentages
            r'increase|decrease|improve|reduce',  # Comparative metrics
            r'measure|metric|score|rating',  # Measurement terms
            r'complete|finish|achieve',  # Completion indicators
            r'all|every|each',  # Totality indicators
        ]

        for criterion in criteria:
            if any(re.search(pattern, criterion.lower()) for pattern in quantifiable_patterns):
                quantifiable_count += 1

        return quantifiable_count / len(criteria) if criteria else 0.0

    def _evaluate_clarity(self, criteria: list[str]) -> float:
        """Evaluate clarity of success criteria."""
        clarity_scores = []

        for criterion in criteria:
            score = 0.0

            # Check for action verbs
            action_verbs = ["identify", "complete", "achieve", "demonstrate",
                          "verify", "validate", "confirm"]
            if any(verb in criterion.lower() for verb in action_verbs):
                score += 0.4

            # Check for specificity (length)
            if len(criterion.split()) >= 5:
                score += 0.3

            # Check for conditional structure
            if any(word in criterion.lower() for word in ["if", "when", "after", "once"]):
                score += 0.3

            clarity_scores.append(score)

        return sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0

    def _evaluate_achievability(self, criteria: list[str]) -> float:
        """Evaluate if success criteria are realistically achievable."""
        achievability_scores = []

        unrealistic_terms = ["perfect", "complete", "all", "every",
                            "100%", "zero", "none", "eliminate"]

        for criterion in criteria:
            # Start with assumption of achievability
            score = 1.0

            # Penalize unrealistic absolute terms
            unrealistic_count = sum(1 for term in unrealistic_terms if term in criterion.lower())
            score -= unrealistic_count * 0.2

            # Boost for phased or gradual criteria
            if any(word in criterion.lower() for word in ["phase", "step", "gradual", "progress"]):
                score = min(1.0, score + 0.2)

            achievability_scores.append(max(0.0, score))

        return (sum(achievability_scores) / len(achievability_scores)
                if achievability_scores else 0.0)


class TemporalGeographicScopeEvaluator(Evaluator):
    """
    Evaluates the appropriateness of temporal and geographic scope definitions.

    Metrics:
    - Temporal boundaries relevance
    - Geographic scope necessity
    - Scope constraint consistency
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate temporal and geographic scope."""
        output: TransformedQuery = ctx.output

        scores = {
            "temporal_relevance": self._evaluate_temporal_scope(output),
            "geographic_relevance": self._evaluate_geographic_scope(output),
            "scope_consistency": self._evaluate_scope_consistency(output)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_temporal_scope(self, output: TransformedQuery) -> float:
        """Evaluate temporal scope appropriateness."""
        # Check if any queries have temporal context
        queries_with_temporal = [q for q in output.search_queries.queries if q.temporal_context]

        # Check for temporal indicators in original query
        temporal_keywords = ["recent", "latest", "current", "historical", "past", "future",
                           "year", "month", "date", "when", "timeline", "trend"]
        needs_temporal = any(keyword in output.original_query.lower()
                            for keyword in temporal_keywords)

        if needs_temporal:
            if queries_with_temporal:
                # Good - temporal context provided when needed
                # Check quality of temporal context
                quality_score = 0.0
                for query in queries_with_temporal:
                    tc = query.temporal_context
                    if tc.start_date or tc.end_date:
                        quality_score += 0.4
                    if tc.recency_preference:
                        quality_score += 0.3
                    if tc.historical_context:
                        quality_score += 0.3

                return min(1.0, quality_score / len(queries_with_temporal))
            else:
                return 0.3  # Needed but not provided
        else:
            if queries_with_temporal:
                return 0.6  # Provided but possibly not needed
            else:
                return 1.0  # Not needed and not provided

    def _evaluate_geographic_scope(self, output: TransformedQuery) -> float:
        """Evaluate geographic scope relevance."""
        # Check for geographic indicators
        geo_keywords = ["country", "region", "global", "local", "national", "international",
                       "location", "where", "area", "city", "state"]
        needs_geographic = any(keyword in output.original_query.lower() for keyword in geo_keywords)

        # Check if scope is defined
        has_scope = bool(output.research_plan.scope_definition)

        if needs_geographic:
            if has_scope:
                # Check if scope mentions geographic boundaries
                if any(geo in output.research_plan.scope_definition.lower()
                       for geo in geo_keywords):
                    return 1.0
                else:
                    return 0.6  # Has scope but not geographic
            else:
                return 0.3  # Needed but not provided
        else:
            if (has_scope and
                any(geo in output.research_plan.scope_definition.lower()
                    for geo in geo_keywords)):
                return 0.7  # Geographic scope when not clearly needed
            else:
                return 1.0  # Appropriately no geographic scope

    def _evaluate_scope_consistency(self, output: TransformedQuery) -> float:
        """Evaluate consistency of scope across transformation."""
        # Check if constraints align with scope
        constraints = output.research_plan.constraints
        scope = output.research_plan.scope_definition

        if not scope and not constraints:
            return 1.0  # No scope or constraints is fine

        consistency_score = 0.5  # Base score

        if scope:
            # Check if scope is reflected in objectives
            scope_terms = set(scope.lower().split())
            objective_text = " ".join(
                obj.objective.lower() for obj in output.research_plan.objectives
            )

            if any(term in objective_text for term in scope_terms if len(term) > 4):
                consistency_score += 0.25

        if constraints:
            # Check if constraints are reasonable
            if len(constraints) <= 5:
                consistency_score += 0.25
            else:
                consistency_score += 0.1  # Too many constraints

        return consistency_score


class SearchSourceSelectionEvaluator(Evaluator):
    """
    Evaluates the appropriateness of search source selections.

    Metrics:
    - Source diversity appropriateness
    - Source-query alignment
    - Domain-specific source usage
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate search source selection."""
        output: TransformedQuery = ctx.output

        scores = {
            "source_diversity": self._evaluate_source_diversity(output),
            "source_alignment": self._evaluate_source_alignment(output),
            "domain_appropriateness": self._evaluate_domain_sources(output)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_source_diversity(self, output: TransformedQuery) -> float:
        """Evaluate diversity of search sources."""
        all_sources = []
        for query in output.search_queries.queries:
            all_sources.extend(query.search_sources)

        if not all_sources:
            return 0.5  # No specific sources selected (neutral)

        unique_sources = set(all_sources)
        diversity_ratio = len(unique_sources) / len(all_sources)

        # Ideal is some diversity but not every query having different sources
        if 0.3 <= diversity_ratio <= 0.7:
            return 1.0
        elif 0.2 <= diversity_ratio < 0.3 or 0.7 < diversity_ratio <= 0.8:
            return 0.7
        else:
            return 0.4

    def _evaluate_source_alignment(self, output: TransformedQuery) -> float:
        """Evaluate if sources align with query types."""
        alignment_scores = []

        for query in output.search_queries.queries:
            if not query.search_sources:
                continue

            score = 0.0
            sources_str = " ".join(str(s) for s in query.search_sources)

            # Check alignment based on query type
            if query.query_type == "factual" and "academic" in sources_str:
                score = 0.9
            elif (query.query_type == "analytical" and
                  ("academic" in sources_str or "technical" in sources_str)):
                score = 0.9
            elif query.query_type == "exploratory" and "web_general" in sources_str:
                score = 0.8
            elif query.query_type == "comparative" and len(query.search_sources) > 1:
                score = 0.8
            elif query.query_type == "temporal" and "news" in sources_str:
                score = 0.9
            else:
                score = 0.5  # Default neutral

            alignment_scores.append(score)

        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5

    def _evaluate_domain_sources(self, output: TransformedQuery) -> float:
        """Evaluate domain-specific source selection."""
        # Detect domain from query
        query_lower = output.original_query.lower()

        domain_source_map = {
            "medical": ["medical_journals", "academic"],
            "health": ["medical_journals", "government"],
            "business": ["industry_reports", "news"],
            "technical": ["technical_docs", "academic"],
            "government": ["government"],
            "social": ["social_media", "news"]
        }

        detected_domains = [domain for domain in domain_source_map if domain in query_lower]

        if not detected_domains:
            return 0.7  # No specific domain detected

        # Check if appropriate sources are used
        expected_sources = set()
        for domain in detected_domains:
            expected_sources.update(domain_source_map[domain])

        actual_sources = set()
        for query in output.search_queries.queries:
            actual_sources.update(str(s) for s in query.search_sources)

        if not actual_sources:
            return 0.3  # Domain detected but no sources specified

        # Check overlap
        matches = sum(1 for exp in expected_sources if any(exp in src for src in actual_sources))
        return min(1.0, matches / len(expected_sources))


class ConfidenceCalibrationEvaluator(Evaluator):
    """
    Evaluates the calibration of confidence scores.

    Metrics:
    - Confidence vs. actual quality correlation
    - Confidence vs. assumption count
    - Confidence vs. gap count
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate confidence calibration."""
        output: TransformedQuery = ctx.output

        scores = {
            "assumption_calibration": self._evaluate_assumption_confidence(output),
            "gap_calibration": self._evaluate_gap_confidence(output),
            "complexity_calibration": self._evaluate_complexity_confidence(output)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_assumption_confidence(self, output: TransformedQuery) -> float:
        """Evaluate if confidence aligns with assumption count."""
        assumption_count = len(output.assumptions_made)
        confidence = output.confidence_score

        # Expected relationship: more assumptions = lower confidence
        if assumption_count == 0:
            expected_confidence = 0.9
        elif assumption_count <= 2:
            expected_confidence = 0.8
        elif assumption_count <= 4:
            expected_confidence = 0.7
        elif assumption_count <= 6:
            expected_confidence = 0.6
        else:
            expected_confidence = 0.5

        # Calculate how well actual confidence matches expected
        deviation = abs(confidence - expected_confidence)

        if deviation <= 0.1:
            return 1.0
        elif deviation <= 0.2:
            return 0.7
        elif deviation <= 0.3:
            return 0.4
        else:
            return 0.2

    def _evaluate_gap_confidence(self, output: TransformedQuery) -> float:
        """Evaluate if confidence aligns with gap count."""
        gap_count = len(output.potential_gaps)
        confidence = output.confidence_score

        # Expected: more gaps = lower confidence
        if gap_count == 0:
            expected_confidence = 0.85
        elif gap_count <= 2:
            expected_confidence = 0.75
        elif gap_count <= 4:
            expected_confidence = 0.65
        else:
            expected_confidence = 0.55

        deviation = abs(confidence - expected_confidence)

        if deviation <= 0.15:
            return 1.0
        elif deviation <= 0.25:
            return 0.6
        else:
            return 0.3

    def _evaluate_complexity_confidence(self, output: TransformedQuery) -> float:
        """Evaluate if confidence aligns with transformation complexity."""
        # Measure complexity
        objective_count = len(output.research_plan.objectives)
        query_count = len(output.search_queries.queries)

        complexity_score = (objective_count + query_count) / 20  # Normalize

        # Higher complexity should have slightly lower confidence
        if complexity_score <= 0.5:
            expected_confidence = 0.85
        elif complexity_score <= 1.0:
            expected_confidence = 0.75
        else:
            expected_confidence = 0.65

        deviation = abs(output.confidence_score - expected_confidence)

        if deviation <= 0.2:
            return 1.0
        elif deviation <= 0.3:
            return 0.6
        else:
            return 0.3


class ExecutionStrategyEvaluator(Evaluator):
    """
    Evaluates the appropriateness of execution strategy selection.

    Metrics:
    - Strategy appropriateness for query batch
    - Dependency handling in HIERARCHICAL mode
    - Parallelization efficiency
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        """Evaluate execution strategy selection."""
        output: TransformedQuery = ctx.output
        batch = output.search_queries

        scores = {
            "strategy_appropriateness": self._evaluate_strategy_choice(batch),
            "dependency_handling": self._evaluate_dependency_handling(output),
            "parallelization": self._evaluate_parallelization(batch)
        }

        return sum(scores.values()) / len(scores)

    def _evaluate_strategy_choice(self, batch) -> float:
        """Evaluate if the chosen strategy is appropriate."""
        strategy = batch.execution_strategy
        query_count = len(batch.queries)

        # Get priority distribution
        priorities = [q.priority for q in batch.queries]
        has_varied_priorities = len(set(priorities)) > 2

        score = 0.0

        if strategy == "SEQUENTIAL":
            # Good for small batches or when order matters
            if query_count <= 5:
                score = 0.9
            else:
                score = 0.4

        elif strategy == "PARALLEL":
            # Good for large batches without dependencies
            if query_count >= 5 and not has_varied_priorities:
                score = 0.9
            elif query_count >= 3:
                score = 0.7
            else:
                score = 0.4

        elif strategy == "ADAPTIVE":
            # Generally a safe choice
            score = 0.8

        elif strategy == "HIERARCHICAL":
            # Good when priorities vary significantly
            if has_varied_priorities:
                score = 1.0
            else:
                score = 0.5

        return score

    def _evaluate_dependency_handling(self, output: TransformedQuery) -> float:
        """Evaluate handling of dependencies between queries."""
        # Check if objectives have dependencies
        has_dependencies = any(obj.dependencies for obj in output.research_plan.objectives)

        if not has_dependencies:
            return 1.0  # No dependencies to handle

        strategy = output.search_queries.execution_strategy

        # HIERARCHICAL or SEQUENTIAL are better for dependencies
        if strategy in ["HIERARCHICAL", "SEQUENTIAL"]:
            # Check if queries are ordered by objective dependencies
            dependency_order = output.research_plan.get_dependency_order()
            dep_obj_ids = [obj.id for obj in dependency_order]

            # Check if high priority queries align with dependency order
            query_order_score = 0.0
            for _i, obj_id in enumerate(dep_obj_ids[:3]):  # Check first 3
                matching_queries = [q for q in output.search_queries.queries
                                  if q.objective_id == obj_id]
                if matching_queries:
                    avg_priority = sum(q.priority for q in matching_queries) / len(matching_queries)
                    if avg_priority <= 2:  # High priority
                        query_order_score += 0.33

            return query_order_score
        else:
            return 0.4  # Not ideal for dependencies

    def _evaluate_parallelization(self, batch) -> float:
        """Evaluate parallelization efficiency."""
        max_parallel = batch.max_parallel
        query_count = len(batch.queries)

        if query_count <= 2:
            # Small batch, parallelization doesn't matter much
            return 1.0

        # Check if max_parallel is reasonable
        if query_count <= 5:
            ideal_parallel = min(3, query_count)
        elif query_count <= 10:
            ideal_parallel = 5
        else:
            ideal_parallel = min(8, query_count // 2)

        deviation = abs(max_parallel - ideal_parallel)

        if deviation == 0:
            return 1.0
        elif deviation <= 2:
            return 0.7
        else:
            return 0.4
