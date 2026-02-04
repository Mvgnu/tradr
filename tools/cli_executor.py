import asyncio
import logging
import os


class GeminiCliRunner:
    """Executes the 'gemini' CLI in a non-interactive subprocess."""

    async def execute(self, prompt: str) -> str:
        command = ["gemini", "-p", prompt, "--model", "gemini-2.5-flash"]
        
        # Set the working directory to the trading folder
        trading_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tradr", "tools", "trading")
        
        logging.info(f"Executing local Gemini CLI command in directory: {trading_dir}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=trading_dir
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=360.0)
            if proc.returncode != 0:
                error_message = stderr.decode().strip()
                logging.error(f"Gemini CLI exited with code {proc.returncode}:\n{error_message}")
                return f"[CLI_EXECUTION_ERROR: {error_message}]"
            return stdout.decode().strip()
        except FileNotFoundError:
            msg = "[CLI_EXECUTION_ERROR: 'gemini' command not found. Please ensure it is installed and in the PATH.]"
            logging.critical(msg)
            return msg
        except asyncio.TimeoutError:
            msg = "[CLI_EXECUTION_ERROR: Command timed out.]"
            logging.error(msg)
            return msg
        except Exception as e:
            msg = f"[CLI_EXECUTION_ERROR: An unexpected error occurred: {e}]"
            logging.critical(msg, exc_info=True)
            return msg
 