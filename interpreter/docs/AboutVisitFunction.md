# About visit function
Visit functions are not reduced to the minimum necessary; the AST tree is not fully known, so more error checking is done for potentially relevant visit functions.

Ex.
```
void IRGenQASM3Visitor::visit(const ASTDelayNode *node) {
  SET_ERROR_INFO("Delay");
}
```
