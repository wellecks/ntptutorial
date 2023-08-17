import Mathlib.Data.Nat.Prime

variable (α: Type) (R S T : Set α)


example (h1: R ⊆ S) (h2: S ⊆ T) : (R ⊆ T) := by
  exact h1.trans h2