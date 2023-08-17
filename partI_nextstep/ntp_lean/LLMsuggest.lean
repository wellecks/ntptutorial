/-
`llmsuggest` tactic for LLM-based next-step suggestions in Lean4.

This is a minimal version of `llmstep` built for tutorial purposes.
  `llmstep`: https://github.com/wellecks/llmstep
-/

import Mathlib.Tactic


open Lean

/- Calls a `suggest.py` python script with the given `args`. -/
def runSuggest (args : Array String) : IO String := do
  let cwd ← IO.currentDir
  let path := cwd / "partI_nextstep" / "ntp_python" / "llmsuggest" / "suggest.py"
  unless ← path.pathExists do
    dbg_trace f!"{path}"
    throw <| IO.userError "could not find python script suggest.py"
  let s ← IO.Process.run { cmd := "python3", args := #[path.toString] ++ args }
  return s

/- Display clickable suggestions in the VSCode Lean Infoview. 
    When a suggestion is clicked, this widget replaces the `llmstep` call 
    with the suggestion, and saves the call in an adjacent comment.
    Code based on `Std.Tactic.TryThis.tryThisWidget`. -/
@[widget] def llmstepTryThisWidget : Widget.UserWidgetDefinition where
  name := "llmstep suggestions"
  javascript := "
import * as React from 'react';
import { EditorContext } from '@leanprover/infoview';
const e = React.createElement;
export default function(props) {
  const editorConnection = React.useContext(EditorContext)
  function onClick(suggestion) {
    editorConnection.api.applyEdit({
      changes: { [props.pos.uri]: [{ range: 
        props.range, 
        newText: suggestion + ' -- ' + props.tactic
        }] }
    })
  }
  return e('div', 
  {className: 'ml1'}, 
  e('ul', {className: 'font-code pre-wrap'}, [
    'Try this: ',
    ...(props.suggestions.map(suggestion => 
        e('li', {onClick: () => onClick(suggestion), 
        className: 'link pointer dim', title: 'Apply suggestion'}, 
        suggestion
      )
    )),
    props.info
  ]))
}"


/- Adds multiple suggestions to the Lean InfoView. 
   Code based on `Std.Tactic.addSuggestion`. -/
def addSuggestions (tacRef : Syntax) (suggestions: List String)
    (origSpan? : Option Syntax := none)
    (extraMsg : String := "") : MetaM Unit := do
  if let some tacticRange := (origSpan?.getD tacRef).getRange? then
    let map ← getFileMap
    let start := findLineStart map.source tacticRange.start
    let body := map.source.findAux (· ≠ ' ') tacticRange.start start
    let texts := suggestions.map fun text => (
      Std.Format.prettyExtra text 
      (indent := (body - start).1) 
      (column := (tacticRange.start - start).1)
    )
    let start := (tacRef.getRange?.getD tacticRange).start
    let stop := (tacRef.getRange?.getD tacticRange).stop
    let stxRange :=
    { start := map.lineStart (map.toPosition start).line
      stop := map.lineStart ((map.toPosition stop).line + 1) }
    let tacticRange := map.utf8RangeToLspRange tacticRange
    let tactic := Std.Format.prettyExtra f!"{tacRef.prettyPrint}"
    let json := Json.mkObj [
      ("tactic", tactic),
      ("suggestions", toJson texts), 
      ("range", toJson tacticRange), 
      ("info", extraMsg)
    ]
    Widget.saveWidgetInfo ``llmstepTryThisWidget json (.ofRange stxRange)


-- `llmsuggest` tactic.
syntax "llmsuggest" : tactic
elab_rules : tactic
  | `(tactic | llmsuggest%$tac) =>
    Lean.Elab.Tactic.withMainContext do
      let goal ← Lean.Elab.Tactic.getMainGoal
      let ppgoal ← Lean.Meta.ppGoal goal
      let ppgoalstr := toString ppgoal
      let suggest ← runSuggest #[ppgoalstr]
      addSuggestions tac $ suggest.splitOn "[SUGGESTION]"
  

/- Examples -/
example : 2 = 2 := by
  rfl -- llmsuggest


example (f : ℕ → ℕ) : Monotone f → ∀ n, f n ≤ f (n + 1) := by
  intro h n -- llmsuggest
  exact h (Nat.le_succ _) -- llmsuggest



