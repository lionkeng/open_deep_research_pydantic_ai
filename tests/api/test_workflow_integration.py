"""Tests for workflow integration fixes."""

from src.core.workflow import ResearchWorkflow, workflow


def test_workflow_singleton_exists():
    """Test that workflow singleton is created."""
    assert workflow is not None
    assert isinstance(workflow, ResearchWorkflow)


def test_workflow_singleton_is_singleton():
    """Test that workflow is truly a singleton."""
    from src.core.workflow import workflow as workflow2
    assert workflow is workflow2


def test_workflow_has_correct_methods():
    """Test that workflow has the expected methods."""
    assert hasattr(workflow, 'run')
    assert hasattr(workflow, 'resume_research')
    assert callable(workflow.run)
    assert callable(workflow.resume_research)

    # Should NOT have execute_research
    assert not hasattr(workflow, 'execute_research')
