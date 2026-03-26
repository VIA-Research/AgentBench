import math
import os
import xml.etree.ElementTree as ET
import requests
import numexpr
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = """Calculate expression using Python's numexpr library.
    Expression should be a single line mathematical expression that solves the problem.
    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    
    def _run(self, expression: str) -> str:
        
        local_dict = {"pi": math.pi, "e": math.e}
        try:  
            return str(
                numexpr.evaluate(
                    expression.strip(),
                    global_dict={},  # restrict access to globals
                    local_dict=local_dict,  # add common mathematical functions
                )
            )
        except KeyboardInterrupt:
            exit(0)
        except:
            return "Error while evaluating expression!"
            
    
    async def _arun(self, expression: str) -> str: 
        local_dict = {"pi": math.pi, "e": math.e}
        return str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )

class WolframAlpha(WolframAlphaAPIWrapper):
    """Subclass of WolframAlphaAPIWrapper with modified run method."""

    _preferred_titles = {"Result", "Results", "Exact result", "Complex roots"}

    def _extract_answers_from_result(self, res: object) -> list[str]:
        try:
            pods = res["pod"]  # type: ignore[index]
        except Exception:
            return []

        if isinstance(pods, dict):
            pods = [pods]
        if not isinstance(pods, list):
            return []

        # Prefer explicit "result-like" pods.
        for pod in pods:
            if not isinstance(pod, dict):
                continue
            if pod.get("@title") not in self._preferred_titles:
                continue
            subpod = pod.get("subpod")
            answers: list[str] = []
            if isinstance(subpod, dict):
                text = str(subpod.get("plaintext", "") or "").strip()
                if text:
                    answers.append(text)
            elif isinstance(subpod, list):
                for sp in subpod:
                    if not isinstance(sp, dict):
                        continue
                    text = str(sp.get("plaintext", "") or "").strip()
                    if text:
                        answers.append(text)
            if answers:
                return answers

        # Fallback to first pod with plaintext.
        for pod in pods:
            if not isinstance(pod, dict):
                continue
            subpod = pod.get("subpod")
            answers: list[str] = []
            if isinstance(subpod, dict):
                text = str(subpod.get("plaintext", "") or "").strip()
                if text:
                    answers.append(text)
            elif isinstance(subpod, list):
                for sp in subpod:
                    if not isinstance(sp, dict):
                        continue
                    text = str(sp.get("plaintext", "") or "").strip()
                    if text:
                        answers.append(text)
            if answers:
                return answers
        return []

    def _direct_query(self, query: str) -> str:
        appid = (self.wolfram_alpha_appid or os.getenv("WOLFRAM_ALPHA_APPID", "")).strip()
        if not appid:
            return "Wolfram Alpha APPID is not configured."

        try:
            resp = requests.get(
                "https://api.wolframalpha.com/v2/query",
                params={"appid": appid, "input": query, "format": "plaintext"},
                timeout=20,
            )
        except Exception as e:
            return f"Wolfram Alpha request failed: {type(e).__name__}: {e}"

        if resp.status_code != 200:
            snippet = (resp.text or "").strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            return (
                f"Wolfram Alpha HTTP error: {resp.status_code}. "
                f"Response: {snippet if snippet else '(empty)'}"
            )

        body = resp.text or ""
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            snippet = body.strip()
            return snippet[:300] if snippet else "Wolfram Alpha returned an unreadable response."

        answers: list[str] = []
        for pod in root.findall("pod"):
            title = pod.attrib.get("title", "")
            if title not in self._preferred_titles:
                continue
            for subpod in pod.findall("subpod"):
                text = (subpod.findtext("plaintext") or "").strip()
                if text:
                    answers.append(text)
            if answers:
                return "\n".join(answers)

        for pod in root.findall("pod"):
            for subpod in pod.findall("subpod"):
                text = (subpod.findtext("plaintext") or "").strip()
                if text:
                    answers.append(text)
            if answers:
                return "\n".join(answers)

        err_msg = root.findtext("./error/msg")
        if err_msg:
            return f"Wolfram Alpha error: {err_msg}"
        return "No good Wolfram Alpha Result was found"

    def run(self, query: str) -> str:
        """Run query through WolframAlpha and parse all results."""
        try:
            res = self.wolfram_client.query(query)
            answers = self._extract_answers_from_result(res)
            if answers:
                return "\n".join(answers)
        except Exception:
            # Some environments/libraries raise AssertionError on non-exact content-type.
            pass
        return self._direct_query(query)

    async def arun(self, query: str) -> str:
        # Keep async API while using the robust sync path.
        return self.run(query)

class WolframAlphaTool(BaseTool):
    name: str = "WolframAlpha"
    description: str = """
    WolframAlpha Tool: Solve complex mathematical equations and perform symbolic computations.  

    Input should be a mathematical expression or equation that WolframAlpha can interpret and solve.  

    Examples:  
        - "Solve 2a + 3 = 2 - 5"  
        - "Factor c^2 + 6c - 27"  
        - "Integrate x^2 from 0 to 1"  
        - "What is 2x+5 = -3x + 7?"

    Use this tool exclusively for advanced mathematical problems, including algebra, calculus, and equation solving.
    If Calculator returns error, try to use WolframAlpha tool to solve complex math question.
    """
    api_wrapper: Optional[WolframAlpha] = None

    def _get_api_wrapper(self) -> WolframAlpha:
        if self.api_wrapper is None:
            self.api_wrapper = WolframAlpha()
        return self.api_wrapper

    def _run(self, query: str) -> str:
        """Run query through WolframAlpha and parse all results."""
        return self._get_api_wrapper().run(query)

    async def _arun(self, query: str) -> str:
        return await self._get_api_wrapper().arun(query)
    
class FinishTool_schema(BaseModel):
    thought: Optional[str] = Field(
        description="Thought of current status and next step based on previous messages."
    )
    answer: str = Field(
        description="""answer should be a value or latex format expression in short without any equations. Do not include any = in your answer. If you can convert the \\frac{{a}}{{b}} into simple values, answer with simplified format.
You must answer with simplified format like as follows:
-12
\\frac{{-6}}{{5}}
(1,2)""",
    )

class FinishTool(BaseTool):
    name: str = "finish"
    description: str = """Finish the question with short answer"""
    # args_schema: Type[BaseModel] = FinishTool_schema
    
    def _run(self, answer: str = "") -> str:
        return f"Answer: {answer}"

    async def _arun(self, answer: str = "") -> str:
        return f"Answer: {answer}"
