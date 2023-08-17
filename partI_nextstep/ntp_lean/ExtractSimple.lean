-- A minimal version of LeanDojo's ExtractData.lean for instructional purposes.
-- Please see LeanDojo's ExtractData.lean for a full working script to use in practice.
-- 
-- Credits: original script is from LeanDojo https://github.com/lean-dojo/LeanDojo/
--     @article{yang2023leandojo,
--      title={{LeanDojo}: Theorem Proving with Retrieval-Augmented Language Models},
--      author={Yang, Kaiyu and Swope, Aidan and Gu, Alex and Chalamala, Rahul and Song, 
--              Peiyang and Yu, Shixing and Godil, Saad and Prenger, Ryan and Anandkumar, Anima},
--      journal={arXiv preprint arXiv:2306.15626},
--      year={2023}
-- }
-- This script is essentially a slightly refactored subset of LeanDojo's script.

import Lean

open Lean Elab System

instance : ToJson Substring where
  toJson s := toJson s.toString

instance : ToJson String.Pos where
  toJson n := toJson n.1

deriving instance ToJson for SourceInfo
deriving instance ToJson for Syntax.Preresolved
deriving instance ToJson for Syntax

structure TacticTrace where
  stateBefore: String
  stateAfter: String
  pos: String.Pos
  endPos: String.Pos
deriving ToJson

structure Trace where
  commandASTs : Array Syntax
  tactics: Array TacticTrace
deriving ToJson

abbrev TraceM := StateT Trace IO


def ppGoals (ctx : ContextInfo) (goals : List MVarId) : IO String :=
  if goals.isEmpty then
    return "no goals"
  else
    let fmt := ctx.runMetaM {} (return Std.Format.prefixJoin "\n\n" (← goals.mapM (Meta.ppGoal ·)))
    return (← fmt).pretty.trim


private def visitTacticInfo (ctx : ContextInfo) (ti : TacticInfo) (parent : InfoTree) : TraceM Unit := do
  match parent with
  | .node (Info.ofTacticInfo i) _ =>
    match i.stx.getKind with
    | `Lean.Parser.Tactic.tacticSeq1Indented | `Lean.Parser.Tactic.tacticSeqBracketed =>
      let ctxBefore := { ctx with mctx := ti.mctxBefore }
      let ctxAfter := { ctx with mctx := ti.mctxAfter }
      let stateBefore ← ppGoals ctxBefore ti.goalsBefore
      let stateAfter ← ppGoals ctxAfter ti.goalsAfter
      let some posBefore := ti.stx.getPos? true | pure ()
      let some posAfter := ti.stx.getTailPos? true | pure ()
      match ti.stx with
      | .node _ _ _ =>
        modifyGet fun trace => ((),
          { trace with tactics := trace.tactics.push { 
            stateBefore := stateBefore, 
            stateAfter := stateAfter, 
            pos := posBefore, 
            endPos := posAfter } }
        )
      | _ => pure ()
    | _ => pure ()
  | _ => pure ()


private def visitInfo (ctx : ContextInfo) (i : Info) (parent : InfoTree) : TraceM Unit := do
  match i with
  | .ofTacticInfo ti => visitTacticInfo ctx ti parent
  | _ => pure ()


private partial def traverseTree (ctx: ContextInfo) (tree : InfoTree) (parent : InfoTree) : TraceM Unit := do
  match tree with
  | .context ctx' t => traverseTree ctx' t tree
  | .node i children =>
    visitInfo ctx i parent
    for x in children do
      traverseTree ctx x tree
  | _ => pure ()


private def traverseTopLevelTree (tree : InfoTree) : TraceM Unit := do
  match tree with
  | .context ctx t => traverseTree ctx t tree
  | _ => throw $ IO.userError "Errors in traverseTopLevelTree; aborting"


def traverseForest (trees : Array InfoTree) : TraceM Trace := do
  for t in trees do
    traverseTopLevelTree t
  get


def relativeTo (path parent : FilePath) : Option FilePath :=
  let rec componentsRelativeTo (pathComps parentComps : List String) : Option FilePath :=
    match pathComps, parentComps with
    | _, [] => mkFilePath pathComps
    | [], _ => none
    | (h₁ :: t₁), (h₂ :: t₂) =>
      if h₁ == h₂ then
        componentsRelativeTo t₁ t₂
      else
        none

    componentsRelativeTo path.components parent.components


def toAbsolute (path : FilePath) : IO FilePath := do
  if path.isAbsolute then
    pure path
  else
    let cwd ← IO.currentDir
    pure $ cwd / path


unsafe def processFile (path : FilePath) : IO Unit := do
  let input ← IO.FS.readFile path
  let opts := Options.empty.setBool `trace.Elab.info true
  enableInitializersExecution
  let inputCtx := Parser.mkInputContext input path.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let (env, messages) ← processHeader header opts messages inputCtx

  if messages.hasErrors then
    for msg in messages.toList do
      if msg.severity == .error then
        println! "ERROR: {← msg.toString}"
    throw $ IO.userError "Errors during import; aborting"

  let some modName := path.fileStem | throw $ IO.userError s!"Invalid path: {path}"
  let env := env.setMainModule modName.toName
  let commandState := { Command.mkState env messages opts with infoState.enabled := true }
  let s ← IO.processCommands inputCtx parserState commandState
  let commands := s.commands.pop -- Remove EOI command.
  let trees := s.commandState.infoState.trees.toArray
  let trace ← (traverseForest trees).run' ⟨#[header] ++ commands, #[]⟩

  let cwd ← IO.currentDir
  let some relativePath := relativeTo path cwd | throw $ IO.userError s!"Invalid path: {path}"
  println! "Input file: {relativePath}"
  let json_path := (
    relativePath.withExtension "ast.json"
  )
  IO.FS.writeFile json_path (toJson trace).pretty
  println! "AST: {json_path}"


unsafe def main (args : List String) : IO Unit := do
  match args with
  | path :: _ =>
    processFile (← toAbsolute ⟨path⟩)
  | [] =>
      println! "Please provide a .lean file (lake env lean --run ExtractData.lean FILENAME.lean)"