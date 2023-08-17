import Mathlib.Data.Nat.Prime

theorem test_thm (m n : Nat) (h : m.coprime n) : m.gcd n = 1 := by 
  rw [Nat.coprime] at h  
  exact h  