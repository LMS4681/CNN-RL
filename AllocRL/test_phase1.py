"""Phase 1 모듈 임포트 및 기본 로직 검증 스크립트."""
import sys
sys.path.insert(0, ".")

from datetime import date

# 1. Calendar 검증
from alloc_env.calendar import (
    is_working_day, get_working_days_between,
    adjust_to_working_day, calculate_end_date, next_working_day
)

# 월요일 = 근무일
assert is_working_day(date(2026, 4, 6))  # 월요일
# 토요일 = 비근무일
assert not is_working_day(date(2026, 4, 4))  # 토요일
# 월~금 = 5일
assert get_working_days_between(date(2026, 4, 6), date(2026, 4, 10)) == 5
# 시작일 포함 3 근무일
assert calculate_end_date(date(2026, 4, 6), 3) == date(2026, 4, 8)
print("[OK] calendar.py")

# 2. Block 검증
from alloc_env.block import Block, PrePlacedBlock

b = Block("TEST", "S001", "BUILD", 10.0, 5.0, 3.0, 50.0,
          date(2026, 4, 6), date(2026, 4, 10))
assert b.length == 10.0
b.turn()
assert b.length == 5.0 and b.breadth == 10.0
b.turn()
assert b.length == 10.0
b2 = b.clone()
assert b2.name == "TEST"
print("[OK] block.py")

# 3. Workspace 검증
from alloc_env.workspace import Workspace, LotRegion

ws = Workspace("WS-01", 0.0, 0.0, 100.0, 200.0)
ws.add_lot(LotRegion("A1", 0.0, 0.0, 50.0, 100.0))
assert len(ws.lots) == 1
assert ws.has_lots
print("[OK] workspace.py")

# 4. Strategy 검증
from alloc_env.strategy import BaseGridStrategy

strat = BaseGridStrategy(step=5.0)
ws.strategy = strat
print("[OK] strategy.py")

# 5. Constraints 검증
from alloc_env.constraints import DimensionConstraint, ValidWorkspacePicker

dc = DimensionConstraint()
assert dc.is_feasible(b, ws)
print("[OK] constraints.py")

# 6. Simulator 검증
from alloc_env.simulator import PlacementSimulator, StagedPipeline, SimulationResult

sim = PlacementSimulator()
print("[OK] simulator.py")

# 7. DataLoader 검증 (임포트만)
from alloc_env.data_loader import load_workspaces, load_blocks
print("[OK] data_loader.py")

print("\n=== All Phase 1 modules verified successfully ===")
