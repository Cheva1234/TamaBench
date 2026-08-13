"""Real-Time Live Benchmark Reporter Dashboard for TamaBench V1.

Uses Rich Live Layout to render real-time streaming updates of simulation clock,
pet/agent state, decision traces, schema compliance, and compute metrics.
"""

from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from tamabench.schemas.observation import Observation
from tamabench.schemas.actions import ActionProposal, StepResult


class LiveReporter:
    def __init__(self, model_name: str = "RuleAgent", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self.console = Console()
        self.live: Optional[Live] = None
        self.decision_history: list[dict] = []
        self.total_decisions: int = 0
        self.valid_schema_count: int = 0
        self.valid_env_count: int = 0
        self.total_income: int = 0
        self.total_spending: int = 0
        self.model_status: str = "idle"
        self.last_observation: Optional[Observation] = None
        self.last_step_result: Optional[StepResult] = None

    def set_model_status(self, status: str):
        """Update display state without implying that an inference is running."""
        if status not in {"idle", "generating"}:
            raise ValueError("status must be 'idle' or 'generating'")
        self.model_status = status
        if self.live:
            self.live.update(self._build_layout(self.last_observation, self.last_step_result))

    def start(self):
        layout = self._build_layout(None, None)
        self.live = Live(layout, console=self.console, refresh_per_second=4)
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()

    def update(
        self,
        observation: Observation,
        step_index: int,
        proposal: Optional[ActionProposal],
        step_result: StepResult,
        latency_ms: float = 0.0,
    ):
        self.last_observation = observation
        self.last_step_result = step_result
        self.total_decisions += 1

        is_schema = 1 if (step_result.error is None or step_result.error.category != "SCHEMA") else 0
        is_env = 1 if step_result.success else 0

        self.valid_schema_count += is_schema
        self.valid_env_count += is_env

        action_str = proposal.action if proposal else "unknown"
        if proposal and proposal.job_id:
            action_str += f"({proposal.job_id})"
        elif proposal and proposal.item:
            action_str += f"({proposal.item})"

        # Record decision event
        self.decision_history.append({
            "step": step_index,
            "day": observation.time.day,
            "time": f"{observation.time.hour:02d}:{observation.time.minute:02d}",
            "action": action_str,
            "schema_valid": is_schema,
            "env_valid": is_env,
            "error": step_result.error.error_type if step_result.error else "-",
            "latency": f"{latency_ms:.1f}ms",
        })

        if len(self.decision_history) > 8:
            self.decision_history.pop(0)

        # Update Economy
        if proposal and proposal.action == "work" and step_result.success:
            j_id = proposal.job_id or ""
            job = next((job for job in observation.jobs_available if job.id == j_id), None)
            if job:
                self.total_income += job.reward

        if proposal and proposal.action == "buy" and step_result.success:
            item = proposal.item or ""
            amt = proposal.amount or 1
            shop_item = next(
                (shop_item for shop_item in observation.shop_items_available if shop_item.item == item),
                None,
            )
            if shop_item:
                self.total_spending += shop_item.cost * amt

        if self.live:
            layout = self._build_layout(observation, step_result)
            self.live.update(layout)

    def _build_layout(self, obs: Optional[Observation], step_res: Optional[StepResult]) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=10),
        )

        # Header Panel
        day_str = f"Day {obs.time.day} {obs.time.hour:02d}:{obs.time.minute:02d}" if obs else "Initializing"
        layout["header"].update(
            Panel(
                f"[bold white]TamaBench V1 Live Monitor[/bold white] | Model: [cyan]{self.model_name}[/cyan] | Seed: #{self.seed} | Sim Time: [yellow]{day_str}[/yellow]",
                style="bold cyan on black",
            )
        )

        # Body Panel: Split between Agent/Pet State and Running Statistics
        layout["body"].split_row(
            Layout(name="pet_agent", ratio=1),
            Layout(name="stats", ratio=1),
        )

        if obs:
            p = obs.pet
            a = obs.agent
            inv = obs.inventory

            # Health bar color
            h_color = "green" if p.health > 50 else ("yellow" if p.health > 25 else "red")

            state_text = (
                f"[bold white]AGENT STATE[/bold white]\n"
                f"• Money: [yellow]${a.money}[/yellow]\n"
                f"• Energy: [cyan]{a.energy}/100[/cyan]\n"
                f"• Activity: [magenta]{a.activity}[/magenta]\n"
                f"• Inventory: Food={inv.food}, Medicine={inv.medicine}\n\n"
                f"[bold white]TAMAGOTCHI STATE[/bold white]\n"
                f"• Health: [{h_color}]{p.health:.1f}/100[/{h_color}]\n"
                f"• Hunger / Fullness: [{'red' if p.hunger < 25 else ('yellow' if p.hunger < 50 else 'green')}]{p.hunger:.1f}/100[/{'red' if p.hunger < 25 else ('yellow' if p.hunger < 50 else 'green')}]\n"
                f"• Energy: {p.energy:.1f}/100\n"
                f"• Happiness: {p.happiness:.1f}/100\n"
                f"• Cleanliness: [{'red' if p.cleanliness < 30 else 'green'}]{p.cleanliness:.1f}/100[/{'red' if p.cleanliness < 30 else 'green'}]\n"
                f"• Sick: [{'red' if p.is_sick else 'green'}]{p.is_sick}[/{'red' if p.is_sick else 'green'}] | Sleeping: {p.is_sleeping}"
            )
            layout["pet_agent"].update(Panel(state_text, title="Live Environment State", border_style="bright_blue"))

            schema_acc = (self.valid_schema_count / self.total_decisions * 100.0) if self.total_decisions > 0 else 100.0
            valid_acc = (self.valid_env_count / self.total_decisions * 100.0) if self.total_decisions > 0 else 100.0
            last_lat = self.decision_history[-1]["latency"] if self.decision_history else "0.0ms"

            stats_text = (
                f"[bold white]MODEL RUNTIME & RESIDENCY[/bold white]\n"
                f"• Residency: [bold green]RESIDENT[/bold green] (RAM/VRAM)\n"
                f"• API Status: [bold cyan]{self.model_status.upper()}[/bold cyan]\n"
                f"• Total API Calls: [white]{self.total_decisions}[/white]\n"
                f"• Last Call Latency: [yellow]{last_lat}[/yellow]\n\n"
                f"[bold white]RUNNING METRICS[/bold white]\n"
                f"• Schema Compliance: [green]{schema_acc:.1f}%[/green]\n"
                f"• Valid Action Rate: [cyan]{valid_acc:.1f}%[/cyan]\n"
                f"• Total Income: [yellow]${self.total_income}[/yellow]\n"
                f"• Total Spending: [magenta]${self.total_spending}[/magenta]"
            )
            layout["stats"].update(Panel(stats_text, title="Real-Time Model & Economy Metrics", border_style="magenta"))

        else:
            layout["pet_agent"].update(Panel("Waiting for environment...", title="State"))
            layout["stats"].update(Panel("Calculating...", title="Metrics"))

        # Footer Panel: Recent Decision Stream Table
        table = Table(border_style="yellow", expand=True)
        table.add_column("Step", style="bold white", width=6)
        table.add_column("Sim Time", style="yellow", width=10)
        table.add_column("Proposed Action", style="cyan")
        table.add_column("Schema", style="green", width=8)
        table.add_column("Valid?", style="magenta", width=8)
        table.add_column("Error", style="red")
        table.add_column("Latency", style="white", width=10)

        for d in self.decision_history:
            sch_str = "[green]✓[/green]" if d["schema_valid"] else "[red]✗[/red]"
            env_str = "[green]✓[/green]" if d["env_valid"] else "[red]✗[/red]"

            table.add_row(
                f"#{d['step']}",
                d["time"],
                d["action"],
                sch_str,
                env_str,
                str(d["error"]),
                d["latency"],
            )

        layout["footer"].update(Panel(table, title="Live Decision Stream", border_style="yellow"))

        return layout
