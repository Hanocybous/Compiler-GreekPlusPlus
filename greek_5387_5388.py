import argparse
from opcode import opmap
from pprint import pprint
from enum import Enum, auto
import time

# Define ANSI color codes
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"  # default color

# Token types
class TokenType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    INTEGER = auto()
    OPERATOR = auto()
    SEPARATOR = auto()
    EOF = auto()

# Token class
class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __str__(self):
        return f'Token({self.type}, {self.value}, {self.line}, {self.column})'
    
    def __repr__(self):
        return self.__str__()

# Lexer class
class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.current_char = self.text[self.pos] if self.text else None

        # List of keywords
        self.keywords = {
            "πρόγραμμα", "δήλωση", "συνάρτηση", "διαδικασία", "αρχή_προγράμματος", "τέλος_προγράμματος",
            "αρχή_συνάρτησης", "τέλος_συνάρτησης", "αρχή_διαδικασίας", "τέλος_διαδικασίας", "επανάλαβε",
            "μέχρι", "όσο", "όσο_τέλος", "για", "για_τέλος", "διάβασε", "γράψε", "εκτέλεσε", "εάν", "τότε",
            "αλλιώς", "εάν_τέλος", "και", "ή", "είσοδος", "έξοδος", "με_βήμα", "διαπροσωπεία", "έως"
        }

        # List of operators
        self.operators = {"+", "-", "*", "/", "=", "<", ">", "<=", ">=", "<>", ":=", "ή", "και", "%"}

        # List of separators
        self.separators = {";", ",", "(", ")", "[", "]", "{", "}", ":"}

    def advance(self):
        """Move to the next character in the input"""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace(self):
        """Skip whitespace characters"""
        while self.current_char is not None and self.current_char.isspace():
            if self.current_char == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.advance()

    def skip_comment(self):
        """Skip over comments"""
        if self.current_char == "{":
            while self.current_char != "}" and self.current_char is not None:
                if self.current_char == "\n":
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.advance()
            if self.current_char == "}":
                self.advance()
        else:
            raise ValueError("Invalid comment syntax")

    def get_integer_token(self):
        """Return an integer token"""
        value = ""
        while self.current_char is not None and self.current_char.isdigit():
            value += self.current_char
            self.advance()
        return Token(TokenType.INTEGER, int(value), self.line, self.column)

    def get_identifier_or_keyword_token(self):
        """Return an identifier or keyword token, enforcing a 30-char limit."""
        value = ""
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            value += self.current_char
            self.advance()

        if len(value) > 30:
            raise ValueError(f"Identifier '{value}' exceeds 30 characters limit.")

        if value in self.keywords:
            return Token(TokenType.KEYWORD, value, self.line, self.column)
        else:
            return Token(TokenType.IDENTIFIER, value, self.line, self.column)


    def get_operator_token(self):
        """Return an operator token, handling multi-character operators like '<=', '>=', '<>'."""
        value = self.current_char
        self.advance()

        if value in {"<", ">"} and self.current_char == "=":
            value += "="
            self.advance()
        elif value == "<" and self.current_char == ">":
            value += ">"
            self.advance()

        return Token(TokenType.OPERATOR, value, self.line, self.column)

    def get_separator_token(self):
        """Return a separator token"""
        value = self.current_char
        self.advance()
        return Token(TokenType.SEPARATOR, value, self.line, self.column)

    def get_next_token(self):
        """Return the next token"""
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            elif self.current_char == "{":
                self.skip_comment()
                continue
            elif self.current_char.isdigit():
                return self.get_integer_token()
            elif self.current_char.isalpha() or self.current_char == "_":
                return self.get_identifier_or_keyword_token()
            elif self.current_char == ":" and self.peek() == "=":
                self.advance()
                self.advance()
                return Token(TokenType.OPERATOR, ":=", self.line, self.column)
            elif self.current_char in self.operators:
                return self.get_operator_token()
            elif self.current_char in self.separators:
                return self.get_separator_token()
            elif self.current_char == "%":
                self.advance()
                return Token(TokenType.OPERATOR, "%", self.line, self.column)
            else:
                raise ValueError(f"Invalid character '{self.current_char}' at line {self.line}, column {self.column}")

        return Token(TokenType.EOF, None, self.line, self.column)

    def peek(self):
        """Peek at the next character without advancing the position"""
        peek_pos = self.pos + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        else:
            return None

    def tokenize(self):
        """Return a list of tokens"""
        tokens = []
        self.current_char = self.text[self.pos]
        while self.current_char is not None:
            token = self.get_next_token()
            tokens.append(token)
        return tokens
        
class SymbolType(Enum):
    VARIABLE = auto()
    CONSTANT = auto()
    FUNCTION = auto()
    PROCEDURE = auto()
    PARAMETER = auto()
    TEMPORARY = auto()

class DataType(Enum):
    INTEGER = "int"
    BOOLEAN = "bool"

class SymbolTableEntry:
    def __init__(self, name, symbol_type, data_type=None, offset=None, scope_level=0, 
                 par_mode=None, value=None, size=None, begin_quad=None, framelength=None,
                 line=None, column=None):
        self.name = name
        self.type = symbol_type
        self.data_type = data_type
        self.par_mode = par_mode
        self.offset = offset
        self.scope_level = scope_level
        self.par_mode = par_mode
        self.value = value
        self.size = size
        self.begin_quad = begin_quad
        self.framelength = framelength
        self.parameter_modes = []
        self.line = line
        self.column = column
        
    def __str__(self):
        info = f"{self.name}: {self.type.name}"
        if self.data_type: 
            type_str = self.data_type.value if isinstance(self.data_type, DataType) else self.data_type
            info += f", type={type_str}"
        if self.offset is not None or self.type == SymbolType.TEMPORARY:
            info += f", offset={self.offset if self.offset is not None else '?'}"
        if self.type == SymbolType.PARAMETER:
            mode = "cv" if self.par_mode == "cv" else "ref"
            info += f", mode={mode}"
        if self.value is not None: 
            info += f", value={self.value}"
        if self.line is not None:
            info += f", declared at line {self.line}"
        
        if self.begin_quad is not None:
            details = []
            details.append(f"firstquad {self.begin_quad}")
            if self.framelength is not None:
                details.append(f"framelength={self.framelength}")
            if self.type in (SymbolType.FUNCTION, SymbolType.PROCEDURE) and self.parameter_modes:
                details.append(f"params: {', '.join(self.parameter_modes)}")
            info += f", {', '.join(details)}"
        
        return info

class SymbolTable:
    RESERVED_NAMES = {
        "πρόγραμμα", "δήλωση", "συνάρτηση", "διαδικασία", "αρχή_προγράμματος", 
        "τέλος_προγράμματος", "αρχή_συνάρτησης", "τέλος_συνάρτησης", 
        "αρχή_διαδικασίας", "τέλος_διαδικασίας", "επανάλαβε", "μέχρι", "όσο", 
        "όσο_τέλος", "για", "για_τέλος", "διάβασε", "γράψε", "εκτέλεσε", "εάν", 
        "τότε", "αλλιώς", "εάν_τέλος", "και", "ή", "είσοδος", "έξοδος", 
        "με_βήμα", "διαπροσωπεία", "ακέραιος", "έως"
    }
    def __init__(self, output_file=None, debug=False):
        self.scopes = [{}]
        self.current_scope_level = 0
        self.offset_counter = 12
        self.offset_stack = [12]
        self.output_file = output_file
        self.debug = debug
        self.snapshot_counter = 1
        self.main_program_name = None
        self.already_warned = set()

        # Clear output file safely
        if self.output_file:
            try:
                with open(self.output_file + ".sym", "w", encoding="utf-8") as file:
                    file.write("")
            except IOError:
                pass  # File might not exist yet
            
    def validate_name(self, name):
        """Check if name is valid (not reserved and follows conventions)"""
        if name in self.RESERVED_NAMES:
            raise ValueError(f"'{name}' is a reserved keyword and cannot be used as an identifier")
        if not name.replace('_', '').isalnum():
            raise ValueError(f"Invalid identifier name '{name}' - only alphanumeric characters and underscores allowed")
        if name[0].isdigit():
            raise ValueError(f"Invalid identifier name '{name}' - cannot start with a digit")

    def check_scope_access(self, name, current_line, current_column):
        """Check if variable is accessible in current scope"""
        entry = self.lookup(name)
        if not entry:
            raise ValueError(f"Undeclared identifier '{name}' at line {current_line}, column {current_column}")
        return entry
    
    def check_shadowing(self, name, line, column):
        """Check if a new declaration shadows an existing variable and issue a warning"""
        existing = self.lookup(name)
        if existing and existing.scope_level < self.current_scope_level:
            # Create a unique identifier for this shadowing case
            warning_id = f"{name}_{line}_{column}"
            
            if warning_id not in self.already_warned:
                self.already_warned.add(warning_id)
                location = f"at line {existing.line}" if existing.line is not None else "in outer scope"
                warning_msg = (
                    f"Variable '{name}' at line {line}, column {column} "
                    f"shadows declaration {location}"
                )
                if self.debug:
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {warning_msg}")
            return True
        return False
    
    def enter_scope(self):
        """Push a new scope onto the stack"""
        self.scopes.append({})
        self.current_scope_level += 1
        # Save current offset before entering new scope
        self.offset_stack.append(self.offset_counter)
        self.offset_counter = 12  # Reset offset counter for new scope

    def exit_scope(self):
        """Pop the current scope from the stack"""
        if len(self.scopes) <= 1:
            return False
            
        self.save_table()
        exited_scope = self.scopes.pop()
        self.current_scope_level -= 1
        # Restore parent scope's offset
        self.offset_counter = self.offset_stack.pop()
        
        # If we're exiting the global scope (scope level 0), update the main program's framelength
        if self.current_scope_level == 0:
            for name, entry in self.scopes[0].items():
                if entry.type == SymbolType.FUNCTION and name == self.main_program_name:
                    entry.framelength = self.offset_counter
        return True

        
    def add_symbol(self, name, symbol_type, line=None, column=None, **kwargs):
        """Add a symbol to the current scope with validation"""
        self.validate_name(name)
        
        if name in self.scopes[-1]:
            raise ValueError(f"Duplicate declaration of '{name}' in the same scope")
            
        # Check for shadowing (just for warning, not prevention)
        self.check_shadowing(name, line, column)
            
        # Handle regular variables and parameters
        if symbol_type in (SymbolType.VARIABLE, SymbolType.PARAMETER, SymbolType.TEMPORARY):
            kwargs['offset'] = self.offset_counter
            self.offset_counter += 4
            
        entry = SymbolTableEntry(
            name, 
            symbol_type,
            line=line,
            column=column,
            **kwargs
        )
        self.scopes[-1][name] = entry
        
        # Track the main program name if it's a function at scope level 0
        if symbol_type == SymbolType.FUNCTION and self.current_scope_level == 0:
            self.main_program_name = name
            
        return entry
        
    def lookup(self, name, current_scope_only=False):
        """Look up a symbol in the symbol table"""
        # Search from current scope outward
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
            if current_scope_only:
                break
        return None
        
    def print_table(self):
        print(f"{Colors.BLUE}[INFO] Symbol Table:{Colors.RESET}")

        # Print current active scopes
        for level, scope in enumerate(self.scopes):
            print(f"{Colors.YELLOW}--- Scope Level {level} ---{Colors.RESET}")
            if not scope:
                print("  (empty)")
            for name, entry in scope.items():
                print(f"  {entry}")
            
        print(f"{Colors.BLUE}[INFO] End of Symbol Table{Colors.RESET}")
        
    def save_table(self):
        """Save the current state of the symbol table to a file, appending each snapshot."""
        if not self.output_file:
            return

        output_file = self.output_file + ".sym"
        try:
            with open(output_file, "a", encoding="utf-8") as file:
                # Write the snapshot with reset levels
                file.write(f"Snapshot {self.snapshot_counter} begin\n")
                for level, scope in enumerate(self.scopes):
                    file.write(f"--- Scope Level {level} ---\n")
                    if not scope:
                        file.write("  (empty)\n")
                    for name, entry in scope.items():
                        if entry.type == SymbolType.TEMPORARY and not self.debug:
                            continue
                        file.write(f"  {entry}\n")
                file.write(f"Snapshot {self.snapshot_counter} end\n\n")  # Add a newline for separation
                self.snapshot_counter += 1  # Increment snapshot counter
        except IOError as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to update symbol table: {e}")
    
    def save_framelength(self):
        if not self.output_file:
            return

        output_file = self.output_file + ".sym"
        try:
            with open(output_file, "a", encoding="utf-8") as file:
                # Write the snapshot with reset levels
                file.write(f"Program Framelength {self.offset_counter}\n")
        except IOError as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to update symbol table: {e}")

class Parser:
    def __init__(self, tokens, filename=None, text=None, show_warnings=False, show_info=False, output_file=None, debug = False):
        self.tokens = tokens
        self.current_token = self.tokens.pop(0) if self.tokens else None
        self.program_name = None
        self.declared_variables = set()
        self.declared_functions = set()
        self.used_variables = set()
        self.filename = filename
        self.text = text
        self.show_warnings = show_warnings
        self.show_info = show_info
        self.symbol_table = SymbolTable(output_file, debug)
        self.code_gen = IntermediateCodeGenerator(self)
        self.current_data_type = "int"
        self.param_modes_stack = []
        self.current_function = None

    EXPECTED_IDENTIFIER_ERROR = "Expected identifier"

    def error(self, message):
        """Raise a syntax error with detailed context"""
        error_message = (
            f"{Colors.RED}[ERROR]{Colors.RESET} {self.filename}:{self.current_token.line}:{self.current_token.column}: {message}\n"
            f"    {self.get_error_context()}\n"
            f"    {' ' * (self.current_token.column - 1)}{Colors.GREEN}^{Colors.RESET}"
        )
        raise SyntaxError(error_message)

    def warning(self, message):
        """Display a warning message with context"""
        # Handle case when there's no current token (post-parsing warnings)
        if self.current_token is None:
            warning_message = (
                f"{Colors.YELLOW}[WARNING]{Colors.RESET} {self.filename}: {message}"
            )
        else:
            warning_message = (
                f"{Colors.YELLOW}[WARNING]{Colors.RESET} {self.filename}:{self.current_token.line}:{self.current_token.column}: {message}\n"
                f"    {self.get_error_context()}\n"
                f"    {' ' * (self.current_token.column - 1)}{Colors.GREEN}^{Colors.RESET}"
            )
        print(warning_message)

    def get_error_context(self):
        """Get the line of code where the error occurred"""
        lines = self.text.splitlines()
        if self.current_token.line - 1 < len(lines):
            return lines[self.current_token.line - 1]
        return ""

    def info(self, message):
        """Display an informational message"""
        print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")

    def check_variable_scope(self, name):
        """Check if variable is properly accessible in current scope"""
        try:
            return self.symbol_table.check_scope_access(
                name, 
                self.current_token.line, 
                self.current_token.column
            )
        except ValueError as e:
            self.error(str(e))

    def check_declared(self, name):
        """Check if an identifier is declared before use"""
        if not self.symbol_table.lookup(name):
            self.error(f"Undeclared identifier '{name}'")

    def check_assignment(self, target, value_type):
        """Check if assignment is type-compatible"""
        target_entry = self.symbol_table.lookup(target)
        if not target_entry:
            return
            
        target_type = target_entry.data_type
        if isinstance(target_type, DataType):
            target_type = target_type.value
            
        if isinstance(value_type, DataType):
            value_type = value_type.value
            
        if target_type != value_type:
            self.error(f"Cannot assign {value_type} to {target_type} variable '{target}'")

    def check_operation(self, left_type, right_type, op):
        """Check if operation is valid for given types"""
        numeric_ops = {'+', '-', '*', '/'}
        comparison_ops = {'=', '<', '>', '<=', '>=', '<>'}
        boolean_ops = {'και', 'ή'}
        
        if op in numeric_ops:
            if left_type == 'int' and right_type == 'int':
                return 'int'
        elif op in comparison_ops:
            if left_type == right_type:
                return 'bool'
        elif op in boolean_ops:
            if left_type == 'bool' and right_type == 'bool':
                return 'bool'
                
        self.error(f"Invalid operation '{op}' between {left_type} and {right_type}")

    def check_type_compatibility(self, type1, type2, operation=None):
        """Check if two types are compatible for an operation"""
        if isinstance(type1, DataType):
            type1 = type1.value
        if isinstance(type2, DataType):
            type2 = type2.value
        
        # Basic type compatibility
        if type1 == type2:
            return True
            
        # Numeric promotions
        if {type1, type2} == {'int'}:
            return True
            
        # Operation-specific checks
        if operation in {'+', '-', '*', '/'}:
            return type1 in {'int'} and type2 in {'int'}
            
        return False

    def check_return_paths(self, func_name):
        """Check that all code paths in a function return a value"""
        func_entry = self.symbol_table.lookup(func_name)
        if not func_entry or func_entry.type != SymbolType.FUNCTION:
            return
        
        # Analyze the quads between begin_quad and end_block
        has_return = False
        for quad in self.code_gen.intermediate_code[func_entry.begin_quad:]:
            if quad[0] == 'retv':
                has_return = True
                break
            elif quad[0] == 'end_block':
                break
        
        if not has_return:
            self.warning(f"Function '{func_name}' may not return a value on all paths")

    def check_variable_initialization(self):
        """Check that variables are initialized before use with proper scope handling"""
        initialized_vars = set()
        
        for quad_index, quad in enumerate(self.code_gen.intermediate_code):
            op, x, y, z = quad
            
            # Skip processing for special quads and temporaries
            if op in ('begin_block', 'end_block', 'halt', 'call', 'par', 'retv', 'input', 'print'):
                continue
                
            # Skip temporary variables (they're managed by the compiler)
            if isinstance(z, str) and z.startswith('T_'):
                continue
                
            # For assignments, mark the target as initialized
            if op == ':=' and isinstance(z, str):
                entry = self.symbol_table.lookup(z)
                if entry and entry.type not in (SymbolType.TEMPORARY, SymbolType.FUNCTION):
                    initialized_vars.add(z)
            
            # For variable references, check initialization
            elif op in ('+', '-', '*', '/'):
                for operand in (x, y):
                    if isinstance(operand, str):
                        # Skip temporaries and function returns
                        if operand.startswith('T_') or operand in self.declared_functions:
                            continue
                            
                        entry = self.symbol_table.lookup(operand)
                        if not entry:
                            continue
                            
                        # Skip warnings for:
                        # 1. Parameters (they're considered initialized)
                        # 2. Global variables (might be initialized later)
                        # 3. Function names
                        if (entry.type == SymbolType.PARAMETER or 
                            entry.scope_level == 0 or
                            entry.type == SymbolType.FUNCTION):
                            continue
                            
                        if operand not in initialized_vars:
                            line_info = f"at quad {quad_index}"
                            self.warning(f"Variable '{operand}' may be used before initialization {line_info}")

    def eat(self, token_type):
        """Consume the current token if it matches the given type"""
        if self.current_token and self.current_token.type == token_type:
            self.current_token = self.tokens.pop(0) if self.tokens else None
        else:
            self.error(f"Expected {token_type}, but found {self.current_token.type if self.current_token else 'EOF'}")
            
    def add_variable(self, name, is_param=False, par_mode=None):
        # Check for shadowing (will issue warning if needed)
        shadowed = self.symbol_table.check_shadowing(name, self.current_token.line, self.current_token.column)
        
        symbol_type = SymbolType.PARAMETER if is_param else SymbolType.VARIABLE
        return self.symbol_table.add_symbol(
            name, 
            symbol_type,
            data_type=self.current_data_type,
            par_mode=par_mode,
            scope_level=self.symbol_table.current_scope_level,
            line=self.current_token.line,
            column=self.current_token.column
        )
        
    def add_temp(self, temp_name):
        """Add a temporary variable and ensure it gets the next offset"""
        return self.symbol_table.add_symbol(
            temp_name,
            SymbolType.TEMPORARY,
            scope_level=self.symbol_table.current_scope_level,
            data_type=self.current_data_type
        )

    def parse_program(self):
        """Parse the program"""
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "πρόγραμμα":
            self.eat(TokenType.KEYWORD)
            if self.current_token.type == TokenType.IDENTIFIER:
                self.program_name = self.current_token.value
                self.eat(TokenType.IDENTIFIER)
                self.parse_programblock()
            else:
                self.error("Expected program name (IDENTIFIER)")
        else:
            self.error("Expected 'πρόγραμμα'")
        self.symbol_table.save_table()
        self.symbol_table.save_framelength()

    def parse_programblock(self):
        """Parse the program block"""
        self.parse_declarations()
        self.parse_subprograms()
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "αρχή_προγράμματος":
            # Get main program entry
            main_entry = self.symbol_table.lookup(self.program_name)
            if main_entry:
                main_entry.begin_quad = self.code_gen.nextquad() + 1
                
            self.code_gen.genquad("begin_block", self.program_name, "_", "_")
            self.eat(TokenType.KEYWORD)
            self.parse_sequence()
            if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "τέλος_προγράμματος":
                self.eat(TokenType.KEYWORD)
                self.code_gen.genquad("halt", "_", "_", "_")
                self.code_gen.genquad("end_block", self.program_name, "_", "_")
                
                # Set framelength for main program
                if main_entry:
                    main_entry.framelength = self.symbol_table.offset_counter

    def parse_declarations(self):
        """Parse variable declarations"""
        while self.current_token.type == TokenType.KEYWORD and self.current_token.value == "δήλωση":
            self.eat(TokenType.KEYWORD)
            self.parse_varlist()
            
    def parse_varlist(self, is_param=False, par_mode=None, add=True):
        if self.current_token.type == TokenType.IDENTIFIER:
            var_name = self.current_token.value
            if add:
                self.declared_variables.add(var_name)
                entry = self.add_variable(var_name, is_param=is_param, par_mode=par_mode)
                if entry:
                    # Set the data type properly
                    entry.data_type = DataType(self.current_data_type) if isinstance(self.current_data_type, str) else self.current_data_type
                
                if is_param:
                    par_modes = self.param_modes_stack.pop()
                    par_modes.append("in" if par_mode == "cv" else "ref")
                    self.param_modes_stack.append(par_modes)
                    
            self.eat(TokenType.IDENTIFIER)
            while self.current_token.type == TokenType.SEPARATOR and self.current_token.value == ",":
                self.eat(TokenType.SEPARATOR)
                if self.current_token.type == TokenType.IDENTIFIER:
                    var_name = self.current_token.value
                    if add:
                        self.declared_variables.add(var_name)
                        self.add_variable(var_name, is_param=is_param, par_mode=par_mode)
                        if (is_param):
                            par_modes = self.param_modes_stack.pop()
                            par_modes.append("in" if par_mode == "cv" else "ref")
                            self.param_modes_stack.append(par_modes)
                    self.eat(TokenType.IDENTIFIER)
                else:
                    self.error("Expected identifier after ',' in variable list.")
        else:
            self.error("Expected identifier in variable declaration.")


    def parse_subprograms(self):
        """Parse subprograms (functions and procedures)"""
        while self.current_token.type == TokenType.KEYWORD and self.current_token.value in {"συνάρτηση", "διαδικασία"}:
            if self.current_token.value == "συνάρτηση":
                self.parse_func()
            elif self.current_token.value == "διαδικασία":
                self.parse_proc()

    def parse_func(self):
        """Parse a function"""
        self.eat(TokenType.KEYWORD)
        if self.current_token.type == TokenType.IDENTIFIER:
            func_name = self.current_token.value
            func_entry = self.symbol_table.add_symbol(
                func_name,
                SymbolType.FUNCTION,
                data_type="int",
                line=self.current_token.line,
                column=self.current_token.column
            )
            
            self.param_modes_stack.append([])
            self.eat(TokenType.IDENTIFIER)
            
            self.symbol_table.enter_scope()
            
            if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "(":
                self.eat(TokenType.SEPARATOR)
                self.parse_formalparlist()
                while self.current_token.type != TokenType.SEPARATOR or self.current_token.value != ")":
                    self.eat(self.current_token.type)
                self.eat(TokenType.SEPARATOR)

            self.parse_funcblock(func_entry, func_name)
        
            func_entry.framelength = self.symbol_table.offset_counter
            func_entry.parameter_modes = self.param_modes_stack.pop()
            
            # Add return paths check here
            self.check_return_paths(func_name)
            
            self.code_gen.genquad("end_block", func_name, "_", "_")
            self.symbol_table.exit_scope()

    def parse_proc(self):
        """Parse a procedure"""
        self.eat(TokenType.KEYWORD)
        if self.current_token.type == TokenType.IDENTIFIER:
            proc_name = self.current_token.value
            proc_entry = self.symbol_table.add_symbol(
                proc_name,
                SymbolType.PROCEDURE,
                line=self.current_token.line,
                column=self.current_token.column
            )
            self.param_modes_stack.append([])
            self.eat(TokenType.IDENTIFIER)

            self.symbol_table.enter_scope()

            if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "(":
                self.eat(TokenType.SEPARATOR)
                self.parse_formalparlist()
                if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == ")":
                    self.eat(TokenType.SEPARATOR)
                else:
                    self.error("Expected ')' after parameter list")

            self.parse_procblock(proc_entry, proc_name)
            
            proc_entry.framelength = self.symbol_table.offset_counter
            proc_entry.parameter_modes = self.param_modes_stack.pop()
            self.code_gen.genquad("end_block", proc_name, "_", "_")
            self.symbol_table.exit_scope()


    def parse_formalparlist(self):
        """Parse the formal parameter list of a function or procedure"""  
        if self.current_token.type == TokenType.IDENTIFIER:
            self.parse_varlist(add=False)

    def parse_funcblock(self, func_entry, func_name):
        """Parse the block of a function"""
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "διαπροσωπεία":
            self.eat(TokenType.KEYWORD)
            self.parse_funcinput()
            self.parse_funcoutput()
            self.parse_declarations()
            self.parse_subprograms()
            if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "αρχή_συνάρτησης":
                func_entry.begin_quad = self.code_gen.nextquad()+1
                # Generate begin_block quad before entering scope
                self.code_gen.genquad("begin_block", func_name, "_", "_")
                self.current_function = func_name
                self.eat(TokenType.KEYWORD)
                self.parse_sequence()
                if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "τέλος_συνάρτησης":
                    self.eat(TokenType.KEYWORD)
                    self.check_return_paths(func_name)
                    self.current_function = None
                else:
                    self.error("Expected 'τέλος_συνάρτησης'")
            else:
                self.error("Expected 'αρχή_συνάρτησης'")
        else:
            self.error("Expected 'διαπροσωπεία'")

    def parse_procblock(self, proc_entry, proc_name):
        """Parse the block of a procedure"""
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "διαπροσωπεία":
            self.eat(TokenType.KEYWORD)
            self.parse_funcinput()
            self.parse_funcoutput()
            self.parse_declarations()
            self.parse_subprograms()
            if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "αρχή_διαδικασίας":
                proc_entry.begin_quad = self.code_gen.nextquad()+1
                self.code_gen.genquad("begin_block", proc_name, "_", "_")
                self.eat(TokenType.KEYWORD)
                self.parse_sequence()
                if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "τέλος_διαδικασίας":
                    self.eat(TokenType.KEYWORD)
                else:
                    self.error("Expected 'τέλος_διαδικασίας'")
            else:
                self.error("Expected 'αρχή_διαδικασίας'")
        else:
            self.error("Expected 'διαπροσωπεία'")

    def parse_funcinput(self):
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "είσοδος":
            self.eat(TokenType.KEYWORD)
            self.parse_varlist(is_param=True, par_mode="cv")

    def parse_funcoutput(self):
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "έξοδος":
            self.eat(TokenType.KEYWORD)
            self.parse_varlist(is_param=True, par_mode="ref")


    def parse_sequence(self):
        """Parse a sequence of statements"""
        self.parse_statement()
        while self.current_token.type == TokenType.SEPARATOR and self.current_token.value == ";":
            self.eat(TokenType.SEPARATOR)
            self.parse_statement()

    def parse_statement(self):
        """Parse a statement"""
        if self.current_token.type == TokenType.KEYWORD:
            if self.current_token.value == "εάν":
                self.parse_if_stat()
            elif self.current_token.value == "όσο":
                self.parse_while_stat()
            elif self.current_token.value == "επανάλαβε":
                self.parse_do_stat()
            elif self.current_token.value == "για":
                self.parse_for_stat()
            elif self.current_token.value == "διάβασε":
                self.parse_input_stat()
            elif self.current_token.value == "γράψε":
                self.parse_print_stat()
            elif self.current_token.value == "εκτέλεσε":
                self.parse_call_stat()
            else:
                self.error("Expected statement")
        elif self.current_token.type == TokenType.IDENTIFIER:
            self.parse_assignment_stat()
        else:
            self.error("Expected statement or assignment")

    def parse_assignment_stat(self):
        """Parse an assignment statement with type checking and constant propagation"""
        if self.current_token.type == TokenType.IDENTIFIER:
            var_name = self.current_token.value
            func_entry = self.symbol_table.lookup(var_name)
            is_function_return = (func_entry and func_entry.type == SymbolType.FUNCTION)

            self.eat(TokenType.IDENTIFIER)
            if self.current_token.type == TokenType.OPERATOR and self.current_token.value == ":=":
                self.eat(TokenType.OPERATOR)
                expr_result = self.parse_expression()
                expr_type = self.get_expression_type(expr_result)
                
                # Enhanced type checking
                if is_function_return:
                    if self.current_function != var_name:
                        self.error("Function return assignment outside function body")

                else:
                    self.check_assignment(var_name, expr_type)
                
                # Update symbol table with constant value if applicable
                #if isinstance(expr_result, (int)):
                #    entry = self.symbol_table.lookup(var_name)
                #    if entry:
                #        entry.value = expr_result
                if hasattr(self, "for_loop_var") and var_name == self.for_loop_var:
                    self.error(f"Loop variable '{var_name}' cannot be modified inside for loop")

                if is_function_return:
                    self.code_gen.genquad(":=", expr_result, "", var_name)
                    self.code_gen.genquad("retv", var_name, "", "_")
                    return
                else:
                    self.code_gen.genquad(":=", expr_result, "", var_name)
            else:
                self.error("Expected ':=' in assignment statement")
        else:
            self.error("Expected identifier in assignment statement")
            
    def parse_expression(self):
        sign = self.parse_optionalsign()
        left = self.parse_term()

        # Check if left is a variable with constant value
        #if isinstance(left, str):
        #    entry = self.symbol_table.lookup(left)
        #    if entry and entry.value is not None:
        #        left = entry.value

        if isinstance(left, (int)):
            left *= sign
        elif sign == -1:
            temp = self.code_gen.newtemp()
            self.code_gen.genquad("*", -1, left, temp)
            left = temp

        left_type = self.get_expression_type(left)

        while self.current_token.type == TokenType.OPERATOR and self.current_token.value in {"+", "-"}:
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.parse_term()

            # Check if right is a variable with constant value
            #if isinstance(right, str):
            #    entry = self.symbol_table.lookup(right)
            #    if entry and entry.value is not None:
            #        right = entry.value

            right_type = self.get_expression_type(right)
            result_type = self.check_operation(left_type, right_type, op)

            # If both operands are constants, compute at compile time
            if isinstance(left, (int)) and isinstance(right, (int)):
                left = left + right if op == "+" else left - right
            else:
                temp = self.code_gen.newtemp()
                self.code_gen.genquad(op, left, right, temp)
                left = temp
                left_type = result_type

        return left

    def get_expression_type(self, expr):
        """Determine the type of an expression with scope checking"""
        if isinstance(expr, int):
            return 'int'
        elif isinstance(expr, str):
            entry = self.check_variable_scope(expr)
            return entry.data_type.value if isinstance(entry.data_type, DataType) else entry.data_type
        return 'int'  # Default type
    
    def parse_term(self):
        left = self.parse_factor()

        # Check if left is a variable with constant value
        #if isinstance(left, str):
        #    entry = self.symbol_table.lookup(left)
        #    if entry and entry.value is not None:
        #        left = entry.value

        while self.current_token.type == TokenType.OPERATOR and self.current_token.value in {"*", "/"}:
            op = self.current_token.value
            self.eat(TokenType.OPERATOR)
            right = self.parse_factor()

            # Check if right is a variable with constant value
            #if isinstance(right, str):
            #    entry = self.symbol_table.lookup(right)
            #    if entry and entry.value is not None:
            #        right = entry.value

            # If both operands are constants, compute at compile time
            if isinstance(left, (int)) and isinstance(right, (int)):
                if op == "*":
                    left = left * right
                else:
                    if right == 0:
                        self.error("Division by zero")
                    left = left // right
            else:
                temp = self.code_gen.newtemp()
                self.code_gen.genquad(op, left, right, temp)
                left = temp

        return left


    def validate_function_call(self, func_name, current_line, current_column):
        """Validate a function call"""
        entry = self.symbol_table.lookup(func_name)
        if not entry:
            raise ValueError(f"Undeclared function '{func_name}' at line {current_line}, column {current_column}")
        if entry.type != SymbolType.FUNCTION:
            raise ValueError(f"'{func_name}' is not a function at line {current_line}, column {current_column}")
        return entry
    
    def parse_factor(self):
        """Parse a factor with semantic analysis"""
        if self.current_token.type == TokenType.INTEGER:
            value = self.current_token.value
            self.eat(TokenType.INTEGER)
            return value
        
        elif self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "(":
            self.eat(TokenType.SEPARATOR)  # Eat '('
            result = self.parse_expression()
            
            if self.current_token.type != TokenType.SEPARATOR or self.current_token.value != ")":
                self.error("Expected ')'")
            self.eat(TokenType.SEPARATOR)  # Eat ')'
            return result
        
        elif self.current_token.type == TokenType.IDENTIFIER:
            id_name = self.current_token.value
            self.eat(TokenType.IDENTIFIER)
            
            # Function call case
            if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "(":
                # Validate function call
                self.validate_function_call(id_name, self.current_token.line, self.current_token.column)
                
                func_entry = self.symbol_table.lookup(id_name)
                if func_entry.type == SymbolType.PROCEDURE:
                    self.error("Procedure cannot be used as an expression")
                if not func_entry:
                    self.error(f"Undeclared function '{id_name}'")
                if func_entry.type not in (SymbolType.FUNCTION, SymbolType.PROCEDURE):
                    self.error(f"'{id_name}' is not callable")
                
                # Generate function call
                temp = None
                if func_entry.type == SymbolType.FUNCTION:
                    temp = self.code_gen.newtemp()
                
                self.parse_actualpars()
                
                if func_entry.type == SymbolType.FUNCTION:
                    self.code_gen.genquad("call", id_name, "_", temp)
                    return temp
                else:
                    self.code_gen.genquad("call", id_name, "_", "_")
                    return 0
            
            # Variable reference case
            else:
                var_entry = self.symbol_table.lookup(id_name)
                if not var_entry:
                    self.error(f"Undeclared variable '{id_name}'")
                if var_entry.type not in (SymbolType.VARIABLE, SymbolType.PARAMETER, SymbolType.TEMPORARY):
                    self.error(f"'{id_name}' is not a variable")
                
                return id_name
        
        else:
            self.error("Expected integer, identifier, or '('")
            return 0

    def parse_optionalsign(self):
        """Parse an optional sign (+ or -)"""
        sign = 1
        if self.current_token.type == TokenType.OPERATOR and self.current_token.value == "-":
            self.eat(TokenType.OPERATOR)
            sign = -1
        elif self.current_token.type == TokenType.OPERATOR and self.current_token.value == "+":
            self.eat(TokenType.OPERATOR)
        return sign

    def parse_idtail(self):
        """Parse the tail of an identifier"""
        if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "(":
            self.parse_actualpars()

    def parse_actualparlist(self):
        """Parse actual parameters list"""
        self.parse_actualparitem()
        while self.current_token.type == TokenType.SEPARATOR and self.current_token.value == ",":
            self.eat(TokenType.SEPARATOR)
            self.parse_actualparitem()

    def parse_actualpars(self):
        """Parse actual parameters"""
        if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "(":
            self.eat(TokenType.SEPARATOR)
            if self.current_token.type != TokenType.SEPARATOR or self.current_token.value != ")":
                self.parse_actualparlist()
            if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == ")":
                self.eat(TokenType.SEPARATOR)
            else:
                self.error("Expected ')'")
        else:
            self.error("Expected '('")

    def parse_actualparitem(self):
        """Parse an actual parameter item with type checking"""
        if self.current_token.type == TokenType.OPERATOR and self.current_token.value == "%":
            self.eat(TokenType.OPERATOR)
            if self.current_token.type == TokenType.IDENTIFIER:
                var_name = self.current_token.value
                if not self.symbol_table.lookup(var_name):
                    self.error("REF parameter must be a variable")
                self.check_declared(var_name)
                self.eat(TokenType.IDENTIFIER)
                self.code_gen.genquad("par", var_name, "REF", "_")
        else:
            expr = self.parse_expression()
            self.code_gen.genquad("par", expr, "CV", "_")

    def parse_condition(self):
        """Parse a full condition with 'ή' (OR) operations."""
        left = self.parse_boolterm()

        while self.current_token.type == TokenType.KEYWORD and self.current_token.value == "ή":
            self.eat(TokenType.KEYWORD)
            right_start = self.code_gen.nextquad()

            right = self.parse_boolterm()

            self.code_gen.backpatch(left.false, right_start)

            left.true = self.code_gen.merge(left.true, right.true)
            left.false = right.false

        return left


    def parse_boolterm(self):
        """Parse a boolterm with 'και' (AND) operations."""
        left = self.parse_boolfactor()

        while self.current_token.type == TokenType.KEYWORD and self.current_token.value == "και":
            self.eat(TokenType.KEYWORD)
            right_start = self.code_gen.nextquad()

            right = self.parse_boolfactor()

            self.code_gen.backpatch(left.true, right_start)

            left.false = self.code_gen.merge(left.false, right.false)
            left.true = right.true

        return left

    def parse_boolfactor(self):
        """Parse a boolean factor"""
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "όχι":
            self.eat(TokenType.KEYWORD)  # Eat 'όχι'
            if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "[":
                self.eat(TokenType.SEPARATOR)  # Eat '['
                condition = self.parse_condition()  # Parse the condition inside the negation
                if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "]":
                    self.eat(TokenType.SEPARATOR)  # Eat ']'
                    # Swap true and false lists for negation
                    return type('Condition', (object,), {'true': condition.false, 'false': condition.true})()
                else:
                    self.error("Expected ']'")
            else:
                self.error("Expected '['")
        elif self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "[":
            self.eat(TokenType.SEPARATOR)  # Eat '['
            condition = self.parse_condition()  # Parse the condition
            if self.current_token.type == TokenType.SEPARATOR and self.current_token.value == "]":
                self.eat(TokenType.SEPARATOR)  # Eat ']'
                return condition  # Return the condition object
            else:
                self.error("Expected ']'")
        else:
            left = self.parse_expression()
            if self.current_token.type == TokenType.OPERATOR and self.current_token.value in {"=", "<=", ">=", "<>", "<", ">"}:
                operator = self.current_token.value
                self.eat(TokenType.OPERATOR)
                right = self.parse_expression()
                
                # Generate comparison with variable names instead of literals
                true_list = self.code_gen.makelist(self.code_gen.nextquad())
                self.code_gen.genquad(operator, left, right, None)
                
                false_list = self.code_gen.makelist(self.code_gen.nextquad())
                self.code_gen.genquad("jump", "_", "_", None)
                
                return type('Condition', (object,), {
                    'true': true_list,
                    'false': false_list
                })()
            else:
                self.error("Expected comparison operator after expression")

    def parse_if_stat(self):
        """Parse an if statement."""
        self.eat(TokenType.KEYWORD)  # eat 'εάν'

        condition = self.parse_condition()

        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "τότε":
            self.eat(TokenType.KEYWORD)

            then_quad = self.code_gen.nextquad()

            self.code_gen.backpatch(condition.true, then_quad)

            self.parse_sequence()

            after_then_quad = self.code_gen.nextquad()

            if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "αλλιώς":
                self.eat(TokenType.KEYWORD)

                jump_after_then = self.code_gen.makelist(self.code_gen.nextquad())
                self.code_gen.genquad("jump", "_", "_", None)

                else_quad = self.code_gen.nextquad()
                self.code_gen.backpatch(condition.false, else_quad)

                self.parse_sequence()

                after_else_quad = self.code_gen.nextquad()
                self.code_gen.backpatch(jump_after_then, after_else_quad)

            else:
                self.code_gen.backpatch(condition.false, after_then_quad)

            if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "εάν_τέλος":
                self.eat(TokenType.KEYWORD)
            else:
                self.error("Expected 'εάν_τέλος'")
        else:
            self.error("Expected 'τότε' after 'εάν'")

    def parse_elsepart(self):
        """Parse the 'else' part of an if statement"""
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "αλλιώς":
            self.eat(TokenType.KEYWORD)
            return self.parse_sequence()
        return None

    def parse_while_stat(self):
        """Parse a while statement with properly set jump targets"""
        self.eat(TokenType.KEYWORD)  # Eat 'όσο'
        
        # Record start of condition for looping back
        start_quad = self.code_gen.nextquad()
        
        # Parse condition - this will now use variables properly
        condition = self.parse_condition()
        
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "επανάλαβε":
            self.eat(TokenType.KEYWORD)
            
            # Backpatch true jumps to loop body start
            body_quad = self.code_gen.nextquad()
            self.code_gen.backpatch(condition.true, body_quad)
            
            # Parse loop body
            self.parse_sequence()
            
            # Jump back to condition
            self.code_gen.genquad("jump", "_", "_", start_quad)
            
            # Backpatch false jumps to exit point
            exit_quad = self.code_gen.nextquad()
            self.code_gen.backpatch(condition.false, exit_quad)
            
            if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "όσο_τέλος":
                self.eat(TokenType.KEYWORD)
            else:
                self.error("Expected 'όσο_τέλος'")
        else:
            self.error("Expected 'επανάλαβε'")

    def parse_do_stat(self):
        """Parse a do-while statement"""
        self.eat(TokenType.KEYWORD)
        start_quad = self.code_gen.nextquad()
        self.parse_sequence()
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "μέχρι":
            self.eat(TokenType.KEYWORD)
            condition = self.parse_condition()
            self.code_gen.backpatch(condition.true, start_quad)
            self.code_gen.backpatch(condition.false, self.code_gen.nextquad())
        else:
            self.error("Expected 'μέχρι'")

    def parse_for_stat(self):
        """Parse a for loop."""
        self.eat(TokenType.KEYWORD)  # eat 'για'
        if self.current_token.type == TokenType.IDENTIFIER:
            loop_var = self.current_token.value
            self.for_loop_var = loop_var
            self.eat(TokenType.IDENTIFIER)

            if self.current_token.type == TokenType.OPERATOR and self.current_token.value == ":=":
                self.eat(TokenType.OPERATOR)
                start_val = self.parse_expression()
                self.code_gen.genquad(":=", start_val, "_", loop_var)

                if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "έως":
                    self.eat(TokenType.KEYWORD)
                    end_val = self.parse_expression()

                    # Default step
                    step_val = "1"
                    if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "με_βήμα":
                        self.eat(TokenType.KEYWORD)
                        step_val = self.parse_expression()

                    # Generate comparison using loop_var instead of literal 1
                    cond_quad = self.code_gen.nextquad()
                    true_list = self.code_gen.makelist(self.code_gen.nextquad())
                    
                    # t = step_val >= 0
                    t_step_pos = self.code_gen.newtemp()
                    self.code_gen.genquad(">=", step_val, 0, t_step_pos)

                    # if step >= 0 goto POS
                    pos_jump = self.code_gen.makelist(self.code_gen.nextquad())
                    self.code_gen.genquad("jump_if_true", t_step_pos, "_", None)

                    # else NEG
                    neg_jump = self.code_gen.makelist(self.code_gen.nextquad())
                    self.code_gen.genquad("jump", "_", "_", None)

                    # POS:
                    pos_quad = self.code_gen.nextquad()
                    self.code_gen.backpatch(pos_jump, pos_quad)

                    true_list_pos = self.code_gen.makelist(self.code_gen.nextquad())
                    self.code_gen.genquad("<=", loop_var, end_val, None)

                    # NEG:
                    neg_quad = self.code_gen.nextquad()
                    self.code_gen.backpatch(neg_jump, neg_quad)

                    true_list_neg = self.code_gen.makelist(self.code_gen.nextquad())
                    self.code_gen.genquad(">=", loop_var, end_val, None)

                    true_list = self.code_gen.merge(true_list_pos, true_list_neg)

                    false_list = self.code_gen.makelist(self.code_gen.nextquad())
                    self.code_gen.genquad("jump", "_", "_", None)

                    if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "επανάλαβε":
                        self.eat(TokenType.KEYWORD)

                        body_start_quad = self.code_gen.nextquad()
                        self.code_gen.backpatch(true_list, body_start_quad)

                        self.parse_sequence()

                        # Step increment
                        tmp = self.code_gen.newtemp()
                        self.code_gen.genquad("+", loop_var, step_val, tmp)
                        self.code_gen.genquad(":=", tmp, "_", loop_var)


                        # Jump to condition again
                        self.code_gen.genquad("jump", "_", "_", cond_quad)

                        exit_quad = self.code_gen.nextquad()
                        self.code_gen.backpatch(false_list, exit_quad)

                        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "για_τέλος":
                            self.eat(TokenType.KEYWORD)
                            del self.for_loop_var
                        else:
                            self.error("Expected 'για_τέλος'")
                    else:
                        self.error("Expected 'επανάλαβε'")
                else:
                    self.error("Expected 'έως'")
            else:
                self.error("Expected ':=' in for statement")
        else:
            self.error("Expected identifier in for statement")

    def parse_step(self):
        """Parse the step part of a for statement"""
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == "με_βήμα":
            self.eat(TokenType.KEYWORD)
            return self.parse_expression()
        return None

    def parse_input_stat(self):
        """Parse an input statement"""
        self.eat(TokenType.KEYWORD)
        if self.current_token.type == TokenType.IDENTIFIER:
            var_name = self.current_token.value
            self.used_variables.add(var_name)
            self.eat(TokenType.IDENTIFIER)
            self.code_gen.genquad("input", "_", "_", var_name)
        else:
            self.error(self.EXPECTED_IDENTIFIER_ERROR)

    def parse_print_stat(self):
        """Parse a print statement."""
        self.eat(TokenType.KEYWORD)
        expr_result = self.parse_expression()
        self.code_gen.genquad("print", expr_result, "_", "_")

    def parse_call_stat(self):
        """Parse a procedure/function call statement."""
        self.eat(TokenType.KEYWORD)
        if self.current_token.type == TokenType.IDENTIFIER:
            func_name = self.current_token.value
            self.eat(TokenType.IDENTIFIER)
            self.parse_idtail()
            self.code_gen.genquad("call", func_name, "_", "_")
        else:
            self.error(self.EXPECTED_IDENTIFIER_ERROR)

    def parse(self):
        self.parse_program()
        #self.code_gen.fold_constants()
        self.code_gen.print_intermediate_code()

class IntermediateCodeGenerator:
    def __init__(self, parser=None):
        self.intermediate_code = []
        self.temp_counter = 0
        self.next_label = 0
        self.parser = parser

    def genquad(self, op, x, y, z):
        """Generate a quadruple with type checking"""
        if not isinstance(op, str):
            raise ValueError("Operator must be a string")
        quad = (op, x, y, z)
        self.intermediate_code.append(quad)
        return len(self.intermediate_code) - 1

    def newtemp(self):
        """Generate a new temporary variable and add it to the symbol table with proper offset"""
        self.temp_counter += 1
        temp_name = f"T_{self.temp_counter}"
        self.parser.add_temp(temp_name)
        return temp_name
        
    def newlabel(self):
        """Generate a new label"""
        self.next_label += 1
        return f"L{self.next_label}"

    def nextquad(self):
        """Get the index of the next quadruple to be generated."""
        return len(self.intermediate_code)

    def backpatch(self, quad_list, target):
        for quad_index in quad_list:
            quad = self.intermediate_code[quad_index]
            if quad[3] is None:  # If the target is not set
                self.intermediate_code[quad_index] = (quad[0], quad[1], quad[2], target)
    
    def save_intermediate_code(self, output_file):
        """Save the intermediate code to a .int file."""
        try:
            with open(output_file, "w", encoding="utf-8") as file:
                for i, quad in enumerate(self.intermediate_code):
                    op, x, y, z = quad
                    file.write(f"{i}: {op}, {x}, {y}, {z}\n")
        except IOError as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to save intermediate code: {e}")
        
    def makelist(self, x):
        """Create a new list containing a single quadruple index."""
        return [x]

    def merge(self, list1, list2):
        """Merge two lists of quadruple indices."""
        return list1 + list2

    def print_intermediate_code(self):
        """Print the intermediate code in a readable format."""
        print(f"{Colors.BLUE}[DEBUG] Intermediate Code:{Colors.RESET}")
        for i, quad in enumerate(self.intermediate_code):
            op, x, y, z = quad
            print(f"{Colors.YELLOW}{i}: {op}, {x}, {y}, {z}{Colors.RESET}")
        print(f"{Colors.BLUE}[DEBUG] Total quads: {len(self.intermediate_code)}{Colors.RESET} \n")
        
    def fold_constants(self):
        """Perform global constant folding and propagation across the IR."""
        changed = True
        while changed:
            changed = False

            const_map = {
                name: entry.value
                for scope in self.parser.symbol_table.scopes
                for name, entry in scope.items()
                if entry.value is not None
            }

            new_quads = []

            def get_const(v):
                if isinstance(v, (int)):
                    return v
                elif isinstance(v, str) and v in const_map:
                    return const_map[v]
                return None

            for op, x, y, z in self.intermediate_code:
                x_val = get_const(x)
                y_val = get_const(y)

                # Constant folding
                if op in {'+', '-', '*', '/'} and x_val is not None and y_val is not None:
                    try:
                        result = {
                            '+': x_val + y_val,
                            '-': x_val - y_val,
                            '*': x_val * y_val,
                            '/': x_val / y_val if y_val != 0 else x_val
                        }[op]
                        new_quads.append((':=', result, '_', z))
                        const_map[z] = result
                        entry = self.parser.symbol_table.lookup(z)
                        if entry:
                            entry.value = result
                        changed = True
                        continue
                    except Exception:
                        pass

                elif op == ':=' and x_val is not None and isinstance(z, str):
                    new_quads.append((':=', x_val, '_', z))
                    const_map[z] = x_val
                    entry = self.parser.symbol_table.lookup(z)
                    if entry:
                        entry.value = x_val
                    changed = True
                    continue

                if isinstance(x, str) and x in const_map:
                    x = const_map[x]
                    changed = True
                if isinstance(y, str) and y in const_map:
                    y = const_map[y]
                    changed = True

                new_quads.append((op, x, y, z))

            self.intermediate_code = new_quads

class RiscvCodeGenerator:
    def __init__(self, intermediate_code, sym_file_path, output_file=None, program_name=None):
        self.intermediate_code = intermediate_code
        self.sym_file_path = sym_file_path
        self.output_file = output_file
        self.program_name = program_name
        self.riscv_code = []
        self.current_function = None
        self.symbol_table, self.program_framelength = self.parse_sym_file()
        self.label_map = {}
        self.temp_registers = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
        self.saved_registers = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
        self.argument_registers = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
        self.current_temp = 0
        self.current_saved = 0
        self.current_arg = 0
        self.used_registers = set()
        self.register_map = {}
        self.spill_offset = 0
        self.index_main_label = 0
        self.main_label = ""
        self.scopes_in_snapshots = []
        self.snapshots = 0
        self.current_snapshot = -1
        self.func_labels = {}
        self.parameter_offset = 0
        self.parameter_stack = []
        
    def parse_sym_file(self):
        """Parse the .sym file to extract symbol table information"""
        symbol_table = {}
        current_snapshot = None
        current_scope = None
        
        symbol_table1 = []
        snapshot = []
        level = []
        
        try:
            with open(self.sym_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Snapshot"):
                        if 'end' in line:
                            snapshot.append(level)
                            level = []
                            symbol_table1.append(snapshot)
                            snapshot = []
                            num_level = 0
                        if "begin" in line:
                            current_snapshot = line.split()[1]
                            symbol_table[current_snapshot] = {}
                        continue
                    elif line.startswith("--- Scope Level"):
                        # "--- Scope Level 0 ---" or "--- Scope Level 1 ---"
                        scope_level_str = line.split("Scope Level")[1].split("---")[0].strip()
                        if(level != []):
                            snapshot.append(level)
                        level = []
                        try:
                            scope_level = int(scope_level_str)
                            if current_snapshot not in symbol_table:
                                symbol_table[current_snapshot] = {}
                            symbol_table[current_snapshot][scope_level] = {}
                            current_scope = scope_level
                        except ValueError:
                            print(f"Warning: Could not parse scope level from line: {line}")
                            continue
                    elif line.startswith("Program Framelength"):
                        framelength = int(line.split()[-1])
                        continue
                    elif line and not line.startswith("Snapshot"):
                        # Parse symbol entry
                        parts = line.split(':', 1)  # Split only on first colon
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            details = parts[1].strip()
                            
                            # Parse symbol details
                            entry = {
                                'name': name,
                                'scope_level': current_scope
                            }
                            
                            # Parse type
                            if 'VARIABLE' in details:
                                entry['type'] = 'VARIABLE'
                            elif 'FUNCTION' in details:
                                entry['type'] = 'FUNCTION'
                            elif 'PROCEDURE' in details:
                                entry['type'] = 'PROCEDURE'
                            elif 'PARAMETER' in details:
                                entry['type'] = 'PARAMETER'
                            elif 'TEMPORARY' in details:
                                entry['type'] = 'TEMPORARY'
                            
                            # Parse data type
                            if 'type=' in details:
                                type_part = details.split('type=')[1].split(',')[0].strip()
                                if 'DataType.' in type_part:
                                    entry['data_type'] = type_part.split('.')[-1].lower()
                                else:
                                    entry['data_type'] = type_part
                            
                            # Parse offset
                            if 'offset=' in details:
                                offset_part = details.split('offset=')[1].split(',')[0].strip()
                                try:
                                    entry['offset'] = int(offset_part)
                                except ValueError:
                                    entry['offset'] = None
                            
                            # Parse parameter mode
                            if 'mode=' in details:
                                mode_part = details.split('mode=')[1].split(',')[0].strip()
                                entry['mode'] = 'cv' if mode_part == 'cv' else 'ref'
                            
                            # Parse value
                            if 'value=' in details:
                                value_part = details.split('value=')[1].split(',')[0].strip()
                                try:
                                    entry['value'] = int(value_part)
                                except ValueError:
                                    entry['value'] = None
                            
                            # Parse firstquad
                            if 'firstquad' in details:
                                firstquad_part = details.split('firstquad')[1].split()[0].strip().strip(',')
                                try:
                                    entry['begin_quad'] = int(firstquad_part)
                                except ValueError:
                                    entry['begin_quad'] = None
                            
                            # Parse framelength
                            if 'framelength=' in details:
                                framelength_part = details.split('framelength=')[1].split(',')[0].strip()
                                try:
                                    entry['framelength'] = int(framelength_part)
                                except ValueError:
                                    entry['framelength'] = None
                            
                            # Parse parameters
                            if 'params:' in details:
                                params_part = details.split('params:')[1].strip()
                                entry['parameter_modes'] = [m.strip() for m in params_part.split(',')]
                            
                            # Parse declared line
                            if 'declared at line' in details:
                                line_part = details.split('declared at line')[-1].split(',')[0].strip()
                                try:
                                    entry['line'] = int(line_part)
                                except ValueError:
                                    entry['line'] = None
                            
                            level.append(entry)
                            pass

        except FileNotFoundError:
            print(f"Error: Symbol table file {self.sym_file_path} not found")
            return {}
        
        # Use the last snapshot as the current symbol table
        if not symbol_table:
            return {}
        
        last_snapshot = sorted(symbol_table.keys())[-1]
        return symbol_table1, framelength
    
    def get_symbol_info(self, name, scope_level=None):
        """Look up a symbol in the symbol table"""
        if scope_level is None:
            # Search all scopes from current outward
            for scope in range(len(self.symbol_table[self.current_snapshot])-1, -1, -1):
                for entry in self.symbol_table[self.current_snapshot][scope]:
                    if name == entry.get('name'):
                       return (entry, scope)            
            
        return (None, None)
    
    def get_next_temp_reg(self):
        """Get a temporary register with better allocation strategy"""
        # Try to find an unused temp register
        for reg in self.temp_registers:
            if reg not in self.used_registers:
                self.used_registers.add(reg)
                return reg
        
        # If no temp registers available, try saved registers
        for reg in self.saved_registers:
            if reg not in self.used_registers:
                self.used_registers.add(reg)
                return reg
        
        # If all registers are used, spill the least recently used one
        return self.spill_register()

    def spill_register(self):
        """Spill a register to memory"""
        for reg in self.temp_registers + self.saved_registers:
            if reg in self.used_registers:
                # Allocate space on stack
                self.spill_offset += 4
                self.emit(f"addi sp, sp, -4")
                self.emit(f"sw {reg}, 0(sp)")
                self.used_registers.remove(reg)
                return reg
        raise RuntimeError("No registers available to spill")
    
    def restore_spilled_registers(self):
        """Restore any spilled registers"""
        if self.spill_offset > 0:
            self.emit(f"addi sp, sp, {self.spill_offset}")
            self.spill_offset = 0

    def free_temp_reg(self, reg):
        """Mark a temporary register as available"""
        if reg in self.used_registers:
            self.used_registers.remove(reg)

    def get_next_saved_reg(self):
        """Get the next available saved register"""
        if self.current_saved >= len(self.saved_registers):
            raise RuntimeError("Out of saved registers")
        reg = self.saved_registers[self.current_saved]
        self.current_saved += 1
        return reg
    
    def get_next_arg_reg(self):
        """Get the next available argument register"""
        if self.current_arg >= len(self.argument_registers):
            raise RuntimeError("Out of argument registers")
        reg = self.argument_registers[self.current_arg]
        self.current_arg += 1
        return reg
    
    def reset_registers(self):
        """Reset register allocation counters"""
        self.current_temp = 0
        self.current_saved = 0
        self.current_arg = 0
    
    def emit(self, code):
        """Emit RISC-V assembly code"""
        self.riscv_code.append(code)
    
    def generate(self):
        """Generate RISC-V code from intermediate code"""
        
        self.collect_label_targets()
        self.map_labels()

        # Generate program header
        self.emit(".text")
        self.emit(".globl " + self.main_label)
        self.emit("")
        
        # Generate data section for strings
        self.emit(".data")
        self.emit("str_nl: .asciz \"\\n\"")
        self.emit("")
        self.emit(".text")
        self.emit("")
        
        self.emit("j " + self.main_label)
        self.emit("")
        
        # Process each quad in intermediate code
        for i, quad in enumerate(self.intermediate_code):
            op, x, y, z = quad
            
            # Check if this quad is a label target
            if i in self.label_targets:
                self.emit("")
                self.emit(f"{self.label_map[i]}:")
                if op == 'begin_block' and x != self.program_name:
                    self.func_labels[x] = self.label_map[i]
                self.emit("")
            
            # Generate code based on operation
            if op == 'begin_block':
                self.current_snapshot += 1
                self.generate_begin_block(x)
            elif op == 'end_block':
                self.generate_end_block(x)
            elif op == ':=':
                self.generate_assignment(x, z)
            elif op in ('+', '-', '*', '/'):
                self.generate_arithmetic(op, x, y, z)
            elif op in ('=', '<', '>', '<=', '>=', '<>'):
                self.generate_comparison(op, x, y, z)
            elif op == 'jump':
                self.generate_jump(z)
            elif op == 'input':
                self.generate_input(z)
            elif op == 'print':
                self.generate_print(x)
            elif op == 'call':
                self.generate_call(x, z)
                self.parameter_stack = []
            elif op == 'par':
                self.parameter_stack.append([x,y])
            elif op == 'halt':
                self.generate_halt()
        
        # Add newline at end
        self.emit("")
        
        # Save to file if specified
        if self.output_file:
            self.save_to_file()
        
        return self.riscv_code
    
    def map_labels(self):
        """Create a mapping from quad indices to sequential labels"""
        label_counter = 0
        for i in range(len(self.intermediate_code)):
            # Only assign labels to quads that are jump targets or begin_block
            if i in self.label_targets:
                if i == self.index_main_label:
                    self.main_label = f"L{label_counter}"
                self.label_map[i] = f"L{label_counter}"
                label_counter += 1
    
    def generate_begin_block(self, func_name):
        """Generate code for begin_block with proper stack frame setup"""
        func_info, sc = self.get_symbol_info(func_name)
        if not func_info:
            if func_name == self.program_name:
                self.emit(f"addi sp, sp, {self.program_framelength}")
                self.emit("mv gp, sp")
            return
            
        self.current_function = func_name
        self.used_registers = set()
        self.spill_offset = 0
        
        if func_name == self.program_name:
            self.emit(f"addi sp, sp, {self.program_framelength}")
            self.emit("mv gp, sp")
        else:
            self.emit("sw ra, (sp)")
        

    def generate_end_block(self, func_name):
        """Generate code for end_block with spill support"""
        func_info, sc = self.get_symbol_info(func_name)
        if not func_info:
            return
            
        # Restore any spilled registers
        self.restore_spilled_registers()
        
        # Function epilogue
        if func_name != self.program_name:
            self.emit("lw ra, (sp)")
            self.emit("jr ra")
            
            
        self.current_function = None
    
    def generate_assignment(self, source, target):
        if source == target:
            return
        src_reg = self.get_next_temp_reg()
        if isinstance(source, int):
            self.emit(f"li {src_reg}, {source}")
        else:
            self.load_value(source, src_reg)
        target_info, sc = self.get_symbol_info(target)
        if target_info:
            if target_info.get('type') == 'FUNCTION':
                self.emit(f"mv a0, {src_reg}")
            elif target_info.get('type') == 'TEMPORARY' or target_info.get('scope_level') == self.get_current_scope_level():
                self.store_value(src_reg, target)
            else:
                self.generate_gnlvcode(target)
                self.emit(f"sw {src_reg}, 0(t0)")
        self.free_temp_reg(src_reg)
    
    def generate_arithmetic(self, op, x, y, z):
        reg1 = self.get_next_temp_reg()
        reg2 = self.get_next_temp_reg()
        result_reg = self.get_next_temp_reg()
        self.load_value(x, reg1)
        self.load_value(y, reg2)
        riscv_op = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
        self.emit(f"{riscv_op} {result_reg}, {reg1}, {reg2}")
        self.store_value(result_reg, z)
        self.free_temp_reg(reg1)
        self.free_temp_reg(reg2)
        self.free_temp_reg(result_reg)
    
    def generate_comparison(self, op, x, y, z):
        """Generate code for comparison operations"""
        reg1 = self.get_next_temp_reg()
        reg2 = self.get_next_temp_reg()
        
        # Load first operand
        if isinstance(x, int):
            self.emit(f"li {reg1}, {x}")
        else:
            x_info, sc = self.get_symbol_info(x)
            if x_info:
                if x_info['type'] == 'TEMPORARY' or x_info['scope_level'] == self.get_current_scope_level():
                    self.load_value(x,reg1)
                else:
                    self.generate_gnlvcode(x)
                    if x_info['type'] == 'PARAMETER' and x_info.get('mode') == 'ref':
                        self.emit(f"lw {reg1}, 0(t0)")
                    else:
                        self.emit(f"lw {reg1}, 0(t0)")
        
        # Load second operand
        if isinstance(y, int):
            self.emit(f"li {reg2}, {y}")
        else:
            y_info, sc = self.get_symbol_info(y)
            if y_info:
                if y_info['type'] == 'TEMPORARY' or y_info['scope_level'] == self.get_current_scope_level():
                    self.emit(f"lw {reg2}, -{y_info['offset']}(fp)")
                else:
                    self.generate_gnlvcode(y)
                    if y_info['type'] == 'PARAMETER' and y_info.get('mode') == 'ref':
                        self.emit(f"lw {reg2}, 0(t0)")
                    else:
                        self.emit(f"lw {reg2}, 0(t0)")
        
        # CASE 1: z is TEMPORARY → boolean result
        if isinstance(z, str) and z.startswith("T_"):
            dest = self.get_next_temp_reg()

            if op == "<":
                self.emit(f"slt {dest}, {reg1}, {reg2}")
            elif op == ">":
                self.emit(f"slt {dest}, {reg2}, {reg1}")
            elif op == "<=":
                self.emit(f"slt {dest}, {reg2}, {reg1}")
                self.emit(f"xori {dest}, {dest}, 1")
            elif op == ">=":
                self.emit(f"slt {dest}, {reg1}, {reg2}")
                self.emit(f"xori {dest}, {dest}, 1")
            elif op == "=":
                self.emit(f"sub {dest}, {reg1}, {reg2}")
                self.emit(f"seqz {dest}, {dest}")
            elif op == "<>":
                self.emit(f"sub {dest}, {reg1}, {reg2}")
                self.emit(f"snez {dest}, {dest}")
            else:
                raise ValueError(f"Unsupported comparison operation: {op}")

            self.store_value(dest, z)
            self.free_temp_reg(dest)

        # CASE 2: z is LABEL → branch
        else:
            label = self.label_map[int(z)]

            if op == "<":
                self.emit(f"blt {reg1}, {reg2}, {label}")
            elif op == ">":
                self.emit(f"bgt {reg1}, {reg2}, {label}")
            elif op == "<=":
                self.emit(f"ble {reg1}, {reg2}, {label}")
            elif op == ">=":
                self.emit(f"bge {reg1}, {reg2}, {label}")
            elif op == "=":
                self.emit(f"beq {reg1}, {reg2}, {label}")
            elif op == "<>":
                self.emit(f"bne {reg1}, {reg2}, {label}")
            else:
                raise ValueError(f"Unsupported comparison operation: {op}")

        
        # Free the registers we used
        self.free_temp_reg(reg1)
        self.free_temp_reg(reg2)
    
    def generate_jump(self, target):
        """Generate unconditional jump"""
        self.emit(f"j {self.label_map[int(target)]}")
    
    def generate_input(self, target):
        """Generate code for input operation"""
        target_info, sc = self.get_symbol_info(target)
        if not target_info:
            return
            
        # System call to read integer
        self.emit("li a7, 5")
        self.emit("ecall")
        
        # Store result
        if target_info['type'] == 'TEMPORARY' or target_info['scope_level'] == self.get_current_scope_level():
            self.emit(f"sw a0, -{target_info['offset']}(fp)")
        else:
            self.generate_gnlvcode(target)
            self.emit("sw a0, (t0)")
    
    def collect_label_targets(self):
        """Collect all quad indices that need labels"""
        self.label_targets = set()
        for i, quad in enumerate(self.intermediate_code):
            op, x, y, z = quad
            # Add jump/branch targets
            if op in ('jump', '=', '<>', '<', '>', '<=', '>='):
                if z is not None and isinstance(z, int):
                    self.label_targets.add(z)
            # Always label begin_block
            if op == 'begin_block':
                if x == self.program_name:
                    self.index_main_label = i
                else:
                    self.func_labels[x] = None
                self.label_targets.add(i)

    def generate_print(self, value):
        """Generate code for print operation"""
        # Get register
        reg = self.get_next_temp_reg()
                
        # Load value to print
        self.load_value(value, reg)
        
        # System call to print integer
        self.emit(f"mv a0, {reg}")
        self.emit("li a7, 1")
        self.emit("ecall")
        
        # Print newline
        self.emit("la a0, str_nl")
        self.emit("li a7, 4")
        self.emit("ecall")
    
    def generate_call(self, func_name, result):
        """Generate code for function call with proper register management"""
        func_info, sc = self.get_symbol_info(func_name)
        if not func_info:
            return
        for par in self.parameter_stack:
            self.generate_parameter(par[0], par[1], func_info.get('framelength'))
            self.parameter_offset+=1
        self.parameter_offset = 0
        
        self.emit(f"sw sp, -4(fp)")
        self.emit(f"addi sp, sp, {func_info.get('framelength')}")
        self.emit(f"jal {self.func_labels.get(func_name)}")
        self.emit(f"addi sp, sp, -{func_info.get('framelength')}")
        if func_info.get('type') == 'FUNCTION':
            result_info, sc = self.get_symbol_info(result)
            if not result_info:
                return
            reg = self.get_next_temp_reg()
            self.emit(f"mv {reg}, a0")
            self.store_value(reg, result)
    
    def generate_parameter(self, value, mode, framelength):
        """Generate code for passing parameters to functions"""
        try:
            reg = self.get_next_temp_reg()
            offset = 12 + 4 * self.parameter_offset    
            if mode == 'CV':
                # Pass by value
                if isinstance(value, int):
                    self.emit(f"li {reg}, {value}")
                else:
                    self.load_value(value, reg)
                self.emit(f"addi fp, sp, {framelength}")
                self.emit(f"sw {reg}, -{offset}(fp)")
                
            elif mode == 'REF':
                # Pass by reference
                value_info, sc = self.get_symbol_info(value)
                if value_info:
                    if value_info.get('scope_level') == self.get_current_scope_level():
                        # Local variable - pass address
                       self.emit(f"addi {reg}, fp, -{value_info['offset']}")
                    else:
                        # Non-local variable
                        self.generate_gnlvcode(value)
                        self.emit(f"mv {reg}, t0")
                    self.emit(f"addi fp, sp, {framelength}")
                    self.emit(f"sw {reg}, -{offset}(fp)")
                    
            elif mode == 'RET':
                # Return value parameter
                value_info, sc = self.get_symbol_info(value)
                if value_info:
                    self.emit(f"addi {reg}, fp, -{value_info['offset']}")
                    self.emit(f"addi sp, sp, -4")
                    self.emit(f"sw {reg}, 0(sp)")
                    
            self.free_temp_reg(reg)
        except RuntimeError as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Register allocation failed during parameter passing")
            raise e

    def generate_halt(self):
        """Generate code for halt operation"""
        self.emit("li a0, 0")
        self.emit("li a7, 93")
        self.emit("ecall")
    
    def generate_gnlvcode(self, var_name):
        """Generate code to get address of non-local variable (gnlvcode)"""
        var_info, sc = self.get_symbol_info(var_name)
        if not var_info:
            return
            
        nesting_diff = self.get_current_scope_level() - var_info['scope_level']
        if nesting_diff < 0:
            raise RuntimeError(f"Invalid nesting level for variable {var_name}")
        
        # Start with current frame pointer
        self.emit("mv t0, sp")
        
        # Follow access links up the static chain
        for _ in range(nesting_diff):
            self.emit("lw t0, -4(t0)")
        
        # Add offset to get variable address
        self.emit(f"addi t0, t0, -{var_info['offset']}")
    
    def load_value(self, value, reg):
        """Load a value (constant or variable) into a register"""
        if isinstance(value, int):
            self.emit(f"li {reg}, {value}")
        else:
            value_info, sc = self.get_symbol_info(value)
            if value_info:
                if sc == 0:
                    self.emit(f"lw {reg}, -{value_info['offset']}(fp)")
                elif value_info['type'] == 'TEMPORARY' or value_info['scope_level'] == self.get_current_scope_level():
                    # Local variable or temporary
                    if 'offset' in value_info:
                        self.emit(f"lw {reg}, -{value_info['offset']}(fp)")
                    elif 'value' in value_info:
                        self.emit(f"li {reg}, {value_info['value']}")
                else:
                    # Non-local variable
                    self.generate_gnlvcode(value)
                    if value_info['type'] == 'PARAMETER' and value_info.get('mode') == 'ref':
                        self.emit(f"lw {reg}, 0(t0)")
                    else:
                        self.emit(f"lw {reg}, 0(t0)")
    
    def store_value(self, reg, target):
        target_info, sc = self.get_symbol_info(target)
        if target_info:
            if target_info.get('type') == 'FUNCTION':
                self.emit(f"mv a0, {reg}")
            elif sc == 0:
                self.emit(f"sw {reg}, -{target_info['offset']}(gp)")
            elif target_info.get('scope_level') == self.get_current_scope_level():
                if 'offset' in target_info:
                    if target_info.get('type') == "PARAMETER":
                        addr_reg = self.get_next_temp_reg()
                        self.load_value(target, addr_reg)
                        self.emit(f"sw {reg}, ({addr_reg})")
                    else:
                        self.emit(f"sw {reg}, -{target_info['offset']}(fp)")
            else:
                self.generate_gnlvcode(target)
                self.emit(f"sw {reg}, 0(t0)")

    def get_current_scope_level(self):
        """Get the current scope level based on current function"""
        if not self.current_function:
            return 0

        func_info, sc = self.get_symbol_info(self.current_function)
        if func_info:
            return func_info['scope_level'] +1 
        return 0
    
    def save_to_file(self):
        """Save generated RISC-V code to file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.riscv_code))
            
        except IOError as e:
            print(f"Error saving RISC-V code: {e}")

def print_help():
    print("Usage: python greek_5387_5388.py <input_file.gr>")
    print("Tokenizes a .gr file written in a custom Greek-like programming language.")
    print("Options:")
    print("  -v, --version   Show the version of the compiler")
    print("  -o, --output    Specify the output file for compilation results")
    print("  -d, --debug     Enable debug mode (prints tokens and other debug info)")
    print("  -p, --print-vars Print declared and used variables")
    print("  -h, --help      Show this help message and exit")
    print("Example: python3 greek_5387_5388.py test.gr")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Greek-like Programming Language Compiler")
    
    # Add arguments
    parser.add_argument(
        "input_file",
        type=str,
        help="Input file to compile (must end with .gr)"
    )
    parser.add_argument(
        "-v", "--version", 
        action="store_true", 
        help="Display the version of the compiler"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        help="Output file to save the compilation results"
    )
    parser.add_argument(
        "-d", "--debug", 
        action="store_true", 
        help="Enable debug mode (prints tokens and other debug info)"
    )
    parser.add_argument(
        "-p", "--print-vars", 
        action="store_true",
        default=False,
        help="Print declared and used variables"
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle version flag
    if args.version:
        print(f"{Colors.GREEN}Greek-like Programming Language Compiler v1.0{Colors.RESET}")
        print(f"{Colors.BLUE}Compilers course project, Spring 2025{Colors.RESET}")
        print(f"{Colors.YELLOW}Authors: Charilaos Chatzidimitriou, AM: 5387{Colors.RESET}")
        print(f"{Colors.YELLOW}Authors: Omiros Chatziiordanis, AM: 5388{Colors.RESET}")
        return

    # Validate the file extension before opening:
    if not args.input_file.lower().endswith(".gr"):
        print(f"{Colors.RED}[ERROR]{Colors.RESET} The input file must end with .gr")
        return

    try:
        with open(args.input_file, "r", encoding="utf-8") as file:
            source_code = file.read()
    except FileNotFoundError:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Input file '{args.input_file}' not found.")
        return

    # Initialize lexer and tokenize the source code
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()

    # Print tokens if debug mode is enabled
    if args.debug:
        print(f"{Colors.BLUE}[DEBUG] Tokens:{Colors.RESET}")
        for token in tokens:
            print(f"{Colors.YELLOW}{token}{Colors.RESET}")
    output_base = args.input_file.rsplit(".", 1)[0]

    # Initialize parser and parse the program
    parser = Parser(tokens, filename=args.input_file, text=source_code, show_warnings=False, show_info=False, output_file=output_base, debug=args.debug)
    try:
        parser.parse_program()
        parser.check_variable_initialization()
    except SyntaxError as e:
        print(f"{Colors.RED}{e}{Colors.RESET}")
        return

    # Print intermediate code if debug mode is enabled
    if args.debug:
        print(f"{Colors.GREEN}[INFO] Compilation successful!{Colors.RESET} \n")
        parser.code_gen.print_intermediate_code()

        # Print the symbol table in the same format as the .sym file
        print(f"{Colors.BLUE}[DEBUG] Symbol Table (as in {output_base}.sym):{Colors.RESET}")
        try:
            with open(output_base + ".sym", "r", encoding="utf-8") as symfile:
                print(f"{Colors.YELLOW}{symfile.read()}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Could not read symbol table file: {e}")

    # Print declared and used variables if the --print-vars flag is set
    if args.print_vars:
        print(f"{Colors.BLUE}[INFO] Declared variables:{Colors.RESET}")
        pprint(parser.declared_variables)
        print(f"{Colors.BLUE}[INFO] Used variables:{Colors.RESET}")
        pprint(parser.used_variables)

    # Save intermediate code to a .int file
    int_file = output_base + ".int"
    parser.code_gen.save_intermediate_code(int_file)

    # Generate and save RISC-V code
    riscv_file = output_base + ".s"
    sym_file = output_base + ".sym"
    riscv_gen = RiscvCodeGenerator(parser.code_gen.intermediate_code, sym_file, riscv_file, parser.program_name)
    riscv_code = riscv_gen.generate()

    if args.debug:
        print(f"{Colors.BLUE}[DEBUG] Generated RISC-V code:{Colors.RESET}")
        for line in riscv_code:
            print(f"{Colors.YELLOW}{line}{Colors.RESET}")

    riscv_gen.save_to_file()

    if args.debug:
        print(f"{Colors.GREEN}[INFO]{Colors.RESET} Intermediate Code code saved to {int_file}")
        print(f"{Colors.GREEN}[INFO]{Colors.RESET} Symbol table saved to {sym_file}")
        print(f"{Colors.GREEN}[INFO]{Colors.RESET} RISC-V code saved to {riscv_file}")

    # Save output to file (if needed)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write("Compilation successful!\n")
            if args.print_vars:
                file.write("\nDeclared variables:\n")
                file.write(f"{parser.declared_variables}\n")
                file.write("\nUsed variables:\n")
                file.write(f"{parser.used_variables}\n")

if __name__ == "__main__":
    main()
    