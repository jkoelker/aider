import os
from typing import AsyncIterator
from contextlib import asynccontextmanager

from aider.main import main as aider_main
from mcp.server.fastmcp import FastMCP, Context
from rich import console

# Prompt templates
FIX_BUG_PROMPT = """Please help me fix this error:

```
{error_message}
```

What might be causing this issue and how can I fix it?"""

IMPLEMENT_FEATURE_PROMPT = """Please help me implement the following feature:

{description}

What files should I modify and what code should I write?"""

EXPLAIN_CODE_PROMPT = """Please explain what this code does:

```
{code}
```

Walk me through the logic step by step."""

REFACTOR_CODE_PROMPT = """Please help me refactor this code:

```
{code}
```

Goal: {goal}

Please suggest improvements to make this code better."""


class AiderContext:
    """Context object for sharing Aider instances between request handlers."""
    def __init__(self, coder):
        self.coder = coder
        self.working_dir = os.getcwd()

    def execute_command(self, command_func, *args, **kwargs):
        """
        Execute a command from Aider's command interface and capture the output.

        Args:
            command_func: The command function to execute
            *args: Arguments to pass to the command function
            **kwargs: Keyword arguments to pass to the command function

        Returns:
            Captured output from the command execution
        """
        if not self.coder:
            return "Error: Aider is not initialized"

        with self.coder.io.console.capture() as capture:
            try:
                result = command_func(*args, **kwargs)
            except Exception as e:
                return f"AIDER Error: {e}"

        try:
            value = capture.get()
        except console.CaptureError as e:
            value = str(e)

        if isinstance(result, str) and result:
            value = result + "\n" + value

        return value


@asynccontextmanager
async def setup_aider_context(server: FastMCP, args=None) -> AsyncIterator[AiderContext]:
    """
    Lifespan manager for Aider MCP server.
    Sets up the Aider coder instance and cleans it up when done.
    """
    yield AiderContext(aider_main(argv=args or [], return_coder=True))


def create_mcp_server(name="Aider", args=None):
    """Create and configure the Aider MCP server."""
    from aider import __version__
    from contextlib import asynccontextmanager

    # Create a proper async context manager for the lifespan
    @asynccontextmanager
    async def lifespan_with_args(server: FastMCP):
        async with setup_aider_context(server, args) as context:
            yield context

    # Create the FastMCP server
    server = FastMCP(
        name=name,
        version=__version__,
        description="AI pair programming in your terminal",
        lifespan=lifespan_with_args,
    )

    # Register all handlers
    register_tools(server)
    register_resources(server)
    register_prompts(server)

    return server


def register_tools(server: FastMCP):
    """Register Aider tools with MCP."""

    @server.tool()
    def aider_chat(message: str, ctx: Context) -> str:
        """
        AI pair programming tool for making targeted code changes. Use this
        tool to:

        1. Implement new features or functionality in existing code
        2. Add tests to an existing codebase
        3. Fix bugs in code
        4. Refactor or improve existing code
        5. Make structural changes across multiple files

        The tool requires a detailed message describing what changes to make.
        Please only describe one change per message.

        Best practices for messages:
          - Be specific about what files or components to modify
          - Describe the desired behavior or functionality clearly
          - Provide context about the existing codebase structure
          - Include any constraints or requirements to follow

        Examples of good messages:
          - Add unit tests for the Customer class in src/models/customer.rb
            testing the validation logic
          - Implement pagination for the user listing API in the
            controllers/users_controller.js file
          - Fix the bug in utils/date_formatter.py where dates before 1970
            aren't handled correctly
          - Refactor the authentication middleware in middleware/auth.js to
            use async/await instead of callbacks
        """
        aider = ctx.request_context.lifespan_context
        aider.execute_command(aider.coder.io.user_input, message)
        aider.execute_command(aider.coder.run_one, message, True)
        return aider.coder.partial_response_content

    @server.tool()
    def aider_add(path: str, ctx: Context) -> str:
        """
        Add files to the chat so aider can edit them or review them in detail.

        This tool lets you:
          - Add individual files by path
          - Add multiple files using glob patterns (like "*.py" or
            "src/**/*.js")
          - Create new files if they don't exist (with confirmation)

        When files are added to the chat, the AI can:
          - See their full content
          - Make changes to them
          - Reference them in responses

        Files added with this tool are editable, unlike those added with
        aider://read-only.

        Examples:
          - "app.py" - Add a specific file
          - "src/*.js" - Add all JavaScript files in the src directory
          - "**/*.css" - Add all CSS files in the project recursively
          - "new_file.txt" - Create and add a new file if it doesn't exist
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_add, path)

    @server.tool()
    def aider_drop(path: str, ctx: Context) -> str:
        """
        Remove files from the chat session to free up context space.

        This tool lets you:
          - Remove specific files by name, freeing up their tokens in the
            context window
          - Drop all files from the chat at once by providing no arguments
          - Use glob patterns to remove multiple files matching a pattern

        Removing files does not delete them from the filesystem; it only
        removes them from the current chat session, making them unavailable
        to the AI for reference or editing.

        Examples:
          - No arguments - Drop all files from the chat session
          - "app.py" - Drop a specific file
          - "*.test.js" - Drop all JavaScript test files
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_drop, path)

    @server.tool()
    def aider_commit(message: str, ctx: Context) -> str:
        """
        Commit all pending changes to the git repository.

        This tool will:
          - Commit all modified files that have been changed in the current
            chat session
          - Commit any other uncommitted changes in the repository
          - Use the provided message as the commit message, or generate one
            automatically

        This is useful after making changes to files via aider to preserve
        those changes in the git history. It's similar to running `git commit`
        manually.

        Examples:
          - No message: Automatically generate a descriptive commit message
          - "Fix database connection bug": Use this as the commit message
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_commit, message)

    @server.tool()
    def aider_run(command: str, ctx: Context) -> str:
        """
        Run a shell command and optionally add the output to the chat.

        This tool allows you to:
          - Execute any shell command in the current repository directory
          - View the output of the command
          - Optionally add the command output to the chat context for the AI
            to analyze

        After running the command, you'll be asked if you want to add the
        output to the chat. If you choose to add it, both the command and its
        output will be included in the chat history.

        This is useful for running tests, checking git status, listing files,
        or any other shell operations that might provide helpful context.

        Examples:
          - "ls -la": List all files in the current directory with details
          - "python -m pytest": Run tests
          - "grep -r \"TODO\" .": Search for TODO comments in the code
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_run, command)

    @server.tool()
    def aider_ls(ctx: Context) -> str:
        """
        List all files in the repository with their chat inclusion status.

        This tool provides a comprehensive view of:
          - Files currently included in the chat for editing (editable files)
          - Read-only files included in the chat for reference
          - Files in the repository that are not currently in the chat

        This helps you understand which files the AI can currently see and
        modify, and which files exist in the project but aren't yet accessible
        to the AI.

        Use this command before adding files to see what's available in the
        repository, or after adding files to confirm what's currently
        accessible to the AI.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_ls, "")

    @server.tool()
    def aider_clear(ctx: Context) -> str:
        """
        Clear the chat history while keeping added files.

        This tool:
          - Removes all previous messages from the chat history
          - Retains all files currently added to the chat (both editable and
            read-only)
          - Frees up token space from previous conversation context

        This is useful when:
          - Starting a new task with the same files
          - The conversation has gotten too long and you want to free up token
            space
          - You want to start fresh while keeping the same files accessible

        Unlike /reset, this command preserves all added files in the chat
        context.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_clear, "")

    @server.tool()
    def aider_reset(ctx: Context) -> str:
        """
        Drop all files and clear the chat history, starting completely fresh.

        This tool performs a complete reset of the current session by:
          - Removing all files from the chat (both editable and read-only)
          - Clearing all previous messages from the chat history
          - Starting with a completely clean slate

        This is the most thorough way to free up context space and start fresh.
        Use this when you want to begin a completely new task with different
        files.

        If you want to keep the current files but clear the chat history,
        use the /clear command instead.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_reset, "")

    @server.tool()
    def aider_tokens(ctx: Context) -> str:
        """
        Report on token usage in the current context window.

        This tool provides a detailed breakdown of token usage by category:
          - System messages (instructions to the AI)
          - Chat history (all previous messages)
          - Repository map (if enabled)
          - Each individual file in the chat
          - Total tokens used and remaining space in the context window

        This helps you understand what's consuming your context window space
        and manage your token usage effectively. If you're approaching the
        token limit, you can use this information to decide what to remove
        with /drop or /clear commands.

        The tool also shows the approximate cost in dollars for the current
        context.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_tokens, "")

    @server.tool()
    def aider_read_only(path: str, ctx: Context) -> str:
        """
        Add files to the chat as read-only references (not for editing).

        This tool lets you:
          - Add files that should be visible to the AI but not editable
          - Add entire directories of files as read-only in one command
          - Convert existing editable files to read-only
          - Add files from outside the git repository for reference

        Read-only files are useful for providing context, documentation, or
        examples without allowing modifications. The AI can see and reference
        these files but cannot modify them.

        Examples:
          - No arguments: Convert all currently editable files to read-only
          - "config.json": Add a specific file as read-only
          - "docs/": Add an entire directory as read-only
          - "/path/to/external/file.txt": Add a file from outside the
            repository
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_read_only, path)

    @server.tool()
    def aider_git(command: str, ctx: Context) -> str:
        """
        Execute git commands in the repository.

        This tool allows you to run any git command directly, with output
        excluded from the chat history. Unlike the run tool, the output is not
        offered to be added to the chat context.

        Use this for git operations like:
          - Checking repository status
          - Viewing commit history
          - Creating branches
          - Viewing diffs
          - Any other git operations

        The command is executed in the repository root directory with the
        GIT_EDITOR environment variable set to prevent interactive editor
        sessions.

        Examples:
          - "status": Show working tree status
          - "log --oneline -n 5": Show recent commit history
          - "branch -a": List all branches
          - "diff": Show uncommitted changes
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_git, command)

    @server.tool()
    def aider_diff(ctx: Context) -> str:
        """
        Display the git diff of changes made since the last message.

        This tool shows all changes made in the repository since the beginning
        of the current conversation, formatted as a git diff. It's useful for:
          - Reviewing what changes have been made so far
          - Checking if changes were applied correctly
          - Getting a summary of all modifications before committing

        The diff shows additions, deletions, and modifications to all tracked
        files, with added lines prefixed with '+' and removed lines with '-'.

        This tool is particularly useful after the AI has suggested changes to
        verify that they were correctly applied to the files.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_diff, "")

    @server.tool()
    def aider_map(ctx: Context) -> str:
        """
        Display the AI's current understanding of the repository structure.

        The repository map is a high-level overview of the codebase that helps
        the AI understand:
          - The overall project structure and organization
          - Important files and directories
          - Relationships between components
          - File types and their purposes

        This map is automatically generated and updated as you work with
        files, and it helps the AI make better suggestions by providing
        broader context about the codebase beyond just the files currently in
        the chat.

        Viewing the map can help you understand what context the AI has about
        your project and whether it has an accurate understanding of the
        codebase structure.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_map, "")

    @server.tool()
    def aider_map_refresh(ctx: Context) -> str:
        """
        Force an immediate regeneration of the repository map.

        This tool triggers an immediate refresh of the repository map, which
        is useful when:
          - New files have been added to the repository outside of aider
          - You've made significant changes to the project structure
          - You want to ensure the AI has the most up-to-date view of the
            codebase

        The repository map is normally refreshed automatically in certain
        situations, but this command lets you force an update at any time.

        After refreshing, use the /map command to view the updated repository
        map.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_map_refresh, "")

    @server.tool()
    def aider_settings(ctx: Context) -> str:
        """
        Display all current Aider configuration settings.

        This tool shows detailed information about the current Aider
        configuration, including:
          - Model settings and configurations
          - Git configuration
          - File handling settings
          - Interface preferences
          - Chat history settings
          - Paths to configuration files
          - Other environment-specific settings

        This is useful for debugging configuration issues, understanding the
        current behavior, or sharing your configuration with others.

        The output includes both command-line settings and those loaded from
        configuration files.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_settings, "")


# NOTE these resources are registered as tools for now, since the sdk
# doesn't support passing the context to resources yet:
# Ref: https://github.com/modelcontextprotocol/python-sdk/issues/244
# Ref: https://github.com/modelcontextprotocol/python-sdk/pull/248
#
# Additionally there is a general bug with resources not being able to
# be templated wtih arguments:
# Ref: https://github.com/modelcontextprotocol/python-sdk/issues/92
def register_resources(server: FastMCP):
    """Register Aider resources as tools with MCP."""

    @server.tool()
    def aider_file(path: str, ctx: Context) -> str:
        """
        Get the raw content of any file in the repository.

        This tool lets you view the content of any file in the repository,
        whether it's currently added to the chat or not. Unlike the add and
        read-only tools, this doesn't add the file to the chat context - it
        just returns the content for immediate viewing.

        This is useful for:
          - Quickly checking a file's content without adding it to the chat
          - Fetching snippets or references from files that don't need
            permanent access
          - Examining files before deciding whether to add them to the chat

        The path can be relative to the repository root or an absolute path.
        """
        aider = ctx.request_context.lifespan_context
        # Convert to absolute path if needed
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        content = aider.coder.io.read_text(path)
        if content is None:
            return f"Error: Could not read {path}"
        return content

    @server.tool()
    def aider_git_status(ctx: Context) -> str:
        """
        Get detailed git status information from the repository.

        This tool runs 'git status' in the repository and returns the output,
        showing:
          - Current branch
          - Tracked files with changes (modified, added, deleted)
          - Untracked files
          - Staged changes ready for commit
          - Relationship to remote branches

        This is useful for understanding the current state of the repository
        and what changes have been made before deciding to commit.

        This is a more direct way to get git status than using aider://git or
        aider://run.
        """
        aider = ctx.request_context.lifespan_context
        return aider.coder.repo.repo.git.status()

    @server.tool()
    def aider_git_diff(ctx: Context) -> str:
        """
        Get the complete git diff of all uncommitted changes.

        This tool runs 'git diff' in the repository and returns the output,
        showing all line-by-line changes that haven't been committed yet. It
        displays:
          - Modified lines with '+' for additions and '-' for deletions
          - Context lines around the changes
          - All files that have been modified

        Unlike aider://diff which shows changes since the start of the
        conversation, this shows all uncommitted changes in the repository
        regardless of when they were made.

        This is useful for reviewing all pending changes before committing or
        to understand the current state of the working directory.
        """
        aider = ctx.request_context.lifespan_context
        return aider.coder.repo.repo.git.diff()

    @server.tool()
    def aider_files_list(ctx: Context) -> str:
        """
        Get a simple list of all files currently in the chat context.

        This tool returns a plain list of all files that are currently added
        to the chat, without any categorization or additional information.
        It only includes files that can be edited (not read-only files).

        This is useful when you need a simple inventory of what editable files
        are currently available to the AI in the chat context.

        For a more comprehensive view of all files including read-only files
        and repository files not in the chat, use the aider://ls tool instead.
        """
        aider = ctx.request_context.lifespan_context
        return "\n".join(aider.coder.get_inchat_relative_files())

    @server.tool()
    def aider_workdir(ctx: Context) -> str:
        """
        Get the current working directory path.

        This tool returns the absolute path to the current working directory
        where aider is running. This is useful for:
          - Understanding file paths in error messages
          - Constructing correct relative paths when adding files
          - Knowing where commands will be executed

        Note that all relative paths in aider are typically interpreted
        relative to the git repository root, which may be different from the
        working directory.

        This tool simply returns the directory path with no additional
        formatting.
        """
        aider = ctx.request_context.lifespan_context
        return aider.working_dir

    @server.tool()
    def aider_web(ctx: Context) -> str:
        """
        Launch the web interface if it's not already running.
        
        This tool opens a browser window with the Aider web interface, allowing you to:
          - Access a browser-based UI for Aider
          - Interact with your code through the web interface
          - Use visual features not available in the terminal interface
        
        If the web server is already running, this will provide the URL to access it.
        If not, it will start the web server and then provide the URL.
        
        No arguments are needed for this command.
        """
        aider = ctx.request_context.lifespan_context
        return aider.execute_command(aider.coder.commands.cmd_web, "")


def register_prompts(server: FastMCP):
    """Register Aider prompts with MCP."""

    @server.prompt()
    def fix_bug(error_message: str) -> str:
        """
        Create a prompt template for debugging an error message.

        This template helps structure a request to fix a bug by providing:
          - A clear error message context
          - A request for analysis of the potential causes
          - A request for solution recommendations

        Simply provide the error message text, and this will return a
        well-structured prompt that you can send to the AI to get assistance
        with debugging.

        This is particularly useful for compiler errors, runtime exceptions,
        test failures, and other error messages that need diagnosis and fixing.
        """
        return FIX_BUG_PROMPT.format(error_message=error_message)

    @server.prompt()
    def implement_feature(description: str) -> str:
        """
        Create a prompt template for implementing a new feature.

        This template structures a request to add a new feature by providing:
          - A clear description of the feature to be implemented
          - A request for file identification (which files need modification)
          - A request for implementation details (what code should be added)

        Simply provide a description of the feature you want to implement,
        and this will return a well-structured prompt that you can send to
        the AI.

        This is useful for both small enhancements and more complex feature
        additions.
        """
        return IMPLEMENT_FEATURE_PROMPT.format(description=description)

    @server.prompt()
    def explain_code(code: str) -> str:
        """
        Create a prompt template for explaining a code snippet.

        This template structures a request to analyze and explain code by:
          - Presenting the code snippet in a clear format
          - Requesting a step-by-step explanation of the logic
          - Focusing on understanding how the code works

        Simply provide the code snippet you want explained, and this will
        return a well-structured prompt that you can send to the AI.

        This is useful for understanding unfamiliar code, legacy code, or
        complex algorithms that need detailed explanation.
        """
        return EXPLAIN_CODE_PROMPT.format(code=code)

    @server.prompt()
    def refactor_code(code: str, goal: str) -> str:
        """
        Create a prompt template for refactoring code.

        This template structures a request to improve existing code by:
          - Presenting the current code that needs refactoring
          - Specifying the goal or purpose of the refactoring
          - Requesting suggestions for code improvements

        Provide both the code snippet to refactor and a clear goal for the
        refactoring, and this will return a well-structured prompt that you
        can send to the AI.

        This is useful for improving code quality, readability, performance,
        or adapting code to new requirements or patterns.
        """
        return REFACTOR_CODE_PROMPT.format(code=code, goal=goal)


def start_server(args=None):
    """
    Start the Aider MCP server using stdio transport.

    Args:
        args: Command line arguments to pass to aider
    """
    server = create_mcp_server(args=args)
    return server.run()
