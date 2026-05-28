"""
Please provide the full URL to your recipes-api GitHub repository below.
"""
import dotenv
import os
from typing import Any
from github import Github, Auth, GithubException
from github.GithubException import UnknownObjectException
import asyncio
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow

dotenv.load_dotenv()

repo_url = "https://github.com/rsCode1/recipes-api.git"  # kept for reference only

git = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN"))) if os.getenv("GITHUB_TOKEN") else Github()

full_repo_name = os.getenv("REPOSITORY")   # GitHub passes this as "username/repo-name"
pr_number = int(os.getenv("PR_NUMBER"))    # GitHub passes the PR number as a string, so cast to int

repo = git.get_repo(full_repo_name)
llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

def get_pr_details(pr_number: int) -> dict | None:
    """
    Fetch metadata for a GitHub pull request.

    Args:
        pr_number: The pull request number.

    Returns:
        A dict with keys: author, title, body, diff_url, state, commit_shas.
        Returns None if the PR does not exist or an API error occurs.
    """
    if not isinstance(pr_number, int) or pr_number <= 0:
        print(f"[get_pr_info] Invalid PR number: {pr_number}")
        return None

    try:
        pr = repo.get_pull(pr_number)
        return {
            "author": pr.user.login,
            "title": pr.title,
            "body": pr.body,
            "diff_url": pr.diff_url,
            "state": pr.state,
            "head_sha": pr.head.sha,
            "commit_shas": [commit.sha for commit in pr.get_commits()],
        }
    except UnknownObjectException:
        print(f"[get_pr_info] PR #{pr_number} not found.")
        return None
    except GithubException as e:
        print(f"[get_pr_info] GitHub API error: {e.status} - {e.data}")
        return None


def get_file_content_from_repo(file_path: str) -> str | None:
    """
    Fetch the decoded text content of a file from the repository.

    Args:
        file_path: Path to the file relative to the repo root (e.g. "src/main.py").

    Returns:
        The file content as a UTF-8 string.
        Returns None if the file does not exist or an API error occurs.
    """
    if not file_path or not file_path.strip():
        print("[get_file_content_from_repo] file_path must be a non-empty string.")
        return None

    try:
        kwargs = {"path": file_path}
        return repo.get_contents(**kwargs).decoded_content.decode("utf-8")
    except UnknownObjectException:
        print(f"[get_file_content_from_repo] File '{file_path}' not found.")
        return None
    except GithubException as e:
        print(f"[get_file_content_from_repo] GitHub API error: {e.status} - {e.data}")
        return None


def pr_commit_details(commit_sha: str) -> list[dict[str, Any]] | None:
    """
    Fetch changed files and patch details for a commit SHA.

    Args:
        commit_sha: A commit SHA from a pull request.

    Returns:
        A list of changed files with filename, status, additions, deletions, changes, and patch.
    """
    if not commit_sha or not commit_sha.strip():
        print("[pr_commit_details] commit_sha must be a non-empty string.")
        return None

    try:
        commit = repo.get_commit(commit_sha)

        changed_files: list[dict[str, Any]] = []

        for f in commit.files:
            changed_files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            })

        return changed_files

    except UnknownObjectException:
        print(f"[pr_commit_details] Commit '{commit_sha}' not found.")
        return None

    except GithubException as e:
        print(f"[pr_commit_details] GitHub API error: {e.status} - {e.data}")
        return None


context_system_prompt ="""
You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
"""

comment_system_prompt = """
You are the commentor agent that writes review comments for pull requests as a human reviewer would.

IMPORTANT RULES:
- You MUST NOT ask the user for information. There is no user to respond.
- You MUST NOT produce any text response before you have context.
- Your FIRST action MUST ALWAYS be to call handoff to ContextAgent. Do this immediately, no exceptions.

Once ContextAgent returns with the gathered context, write a ~200-300 word review in markdown format detailing:
   - What is good about the PR?
   - Did the author follow ALL contribution rules? What is missing?
   - Are there tests for new functionality? If there are new models, are there migrations for them? Use the diff to determine this.
   - Are new endpoints documented? Use the diff to determine this.
   - Which lines could be improved upon? Quote these lines and offer suggestions.

- If you need more details after receiving context, hand off to ContextAgent again.
- Directly address the author e.g. "Thanks for fixing this. Can you roll this fix out everywhere?"
- Once your review is drafted, you MUST call add_draft_comment_to_state and then hand off to ReviewAndPostingAgent.
"""

review_system_prompt = """
You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.  

"""
# ✅ Plain functions — LlamaIndex wraps them automatically
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

async def add_draft_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """add draft comment to state"""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["draft_comment"] = draft_comment
    return "Draft comment saved"

async def add_pr_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """add pr context to state"""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["gathered_contexts"] = gathered_contexts
    return "Context saved"
async def add_final_pr_review_to_state(ctx: Context, final_review: str) -> str:
    """add final review comment to state"""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["final_review"] = final_review
    return "Final review saved"


def post_final_pr_review_to_github(pr_number: int, final_review: str) -> str:
    """post final pr review to github"""
    try:
        pr = repo.get_pull(pr_number)
        pr.create_review(body=final_review, event="COMMENT")
        return f"Review posted successfully to PR #{pr_number}"
    except UnknownObjectException:
        return f"PR #{pr_number} not found."
    except GithubException as e:
        return f"GitHub API error: {e.status} - {e.data}"

get_pr_details_tool = FunctionTool.from_defaults(get_pr_details)
get_file_content_tool = FunctionTool.from_defaults(get_file_content_from_repo)
pr_commit_details_tool = FunctionTool.from_defaults(pr_commit_details)
post_final_pr_review_to_github_tool = FunctionTool.from_defaults(post_final_pr_review_to_github)

add_pr_context_to_state_tool = FunctionTool.from_defaults(add_pr_context_to_state)
add_draft_comment_to_state_tool = FunctionTool.from_defaults(add_draft_comment_to_state)
add_final_pr_review_to_state_tool = FunctionTool.from_defaults(add_final_pr_review_to_state)
# context_agent = ReActAgent(
#     tools=[get_pr_details_tool, get_file_content_tool, pr_commit_details_tool],
#     llm=llm,
#     system_prompt=context_system_prompt,
#     name="ContextAgent",
#     max_iterations=20,   # also add this — 3+ tool calls needed per query
# )

context_agent = FunctionAgent(
	llm=llm,
	name="ContextAgent",
	description="Gathers all the needed context  ",
	tools=[get_pr_details_tool, get_file_content_tool, pr_commit_details_tool,add_pr_context_to_state_tool],
	system_prompt=context_system_prompt,
	can_handoff_to = ["CommentorAgent"]
)
commentor_agent = FunctionAgent(
	llm=llm,
	name="CommentorAgent",
	description="drafts pr comments",
    tools=[add_draft_comment_to_state_tool],
	system_prompt=comment_system_prompt,
	can_handoff_to = ["ContextAgent","ReviewAndPostingAgent"]
)
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="review draft pr comments and post to GitHub",
    tools=[add_final_pr_review_to_state_tool, post_final_pr_review_to_github_tool],
    system_prompt=review_system_prompt,
    can_handoff_to = ["CommentorAgent"]
)
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent,review_and_posting_agent],
    root_agent= review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)
async def main():
    query = f"Write a review for PR number {pr_number}"  
    prompt = RichPromptTemplate(query)
    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
#print (pr_commit_details(get_pr_info(1)["commit_shas"]))
#print(get_file_content_from_repo("README.md"))