
# mlBridgeBiddingLib.py

import gc
import multiprocessing as mp
import pathlib
import pickle
import re
import time
import operator
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import psutil
import tqdm
from asteval import Interpreter  # type: ignore[import-not-found]


# created and maintained by ChatGpt 4.0o on 22-Jun-2024
class CriteriaEvaluator:
    """A class to parse and evaluate mathematical/logical expressions."""
    
    def __init__(self):
        # Define the operator precedence and functions
        # todo: looks like ~ (bitwise not) is missing? Or is 'not' also used?
        self.ops = {
            '==': operator.eq,
            '!=': operator.ne,
            '<=': operator.le,
            '>=': operator.ge,
            '<': operator.lt,
            '>': operator.gt,
            '&': np.logical_and,
            '|': np.logical_or,
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
            'and': np.logical_and,
            'or': np.logical_or,
            'not': np.logical_not
        }
        
        self.precedence = {
            'or': 1,    # Lowest precedence
            '|': 1,     # Same as 'or'
            'and': 2,   # Higher than 'or'
            '&': 2,     # Same as 'and'
            '==': 3, '!=': 3,  # Comparison operators
            '<': 3, '<=': 3, '>': 3, '>=': 3,  
            '+': 4, '-': 4,    # Arithmetic operators
            '*': 5, '/': 5, '//': 5, '%': 5,
            '**': 6,           # Exponentiation
            'not': 7          # Highest precedence
        }
        
        # Regex pattern for tokenizing expressions
        self.token_pattern = r'\d+|<=|>=|==|!=|\b[a-zA-Z_]\w*\b|[&|<>+*/%()-]|\bnot\b|\band\b|\bor\b|//|\*\*'

        self.next_bidding_seat_d = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}

        self.incomplete_rules = set()

        # Initialize attributes for storing criteria and expressions
        self.criteria_expressions_infix_l = []
        self.criteria_variables_infix_sym_to_index_d = {}
        self.criteria_variables_infix_index_to_sym_d = {}
        self.metric_col_d = defaultdict(list)
        self.bt_index_to_metric_col_d = {}
        self.bt_index_to_criteria_exprs_d = {}
        self.bt_index_to_criteria_cols_d = {}
        self.directional_exprs_cols_d = {}
        self.directional_variable_cols_d = {}
        self.regex_col_names_d = {}
    

    def strip_comments(self, expr: str) -> str:
        """Strip comments from an expression.
        
        A '#' character marks the beginning of a comment - everything from '#' 
        to the end of the line is ignored.
        
        Args:
            expr: The expression string, possibly containing comments
            
        Returns:
            The expression with comments removed and whitespace stripped
        """
        # Find the position of '#' and take only the part before it
        comment_pos = expr.find('#')
        if comment_pos != -1:
            expr = expr[:comment_pos]
        return expr.strip()


    def _chained_comparison(self, tokens):
        """Pre-process chained comparisons before converting to postfix."""
        if len(tokens) >= 5:
            # Look for patterns like: a op1 b op2 c
            ops = {'<', '<=', '>', '>=', '==', '!='}
            if tokens[1] in ops and tokens[3] in ops:
                # Convert "a op1 b op2 c" to "(a op1 b) and (b op2 c)"
                a, op1, b, op2, c = tokens[:5]
                return [a, b, op1, c, op2, 'and']
        return tokens


    def infix_to_postfix(self, tokens):
        """Convert infix tokens to postfix notation with proper operator precedence."""
        stack = []
        output = []
        
        for token in tokens:
            
            # Operands go directly to output
            if token.isnumeric() or re.match(r'\b[a-zA-Z_]\w*\b', token):
                output.append(token)
                
            # Operators get processed according to precedence
            elif token in self.precedence:
                # Pop operators with higher/equal precedence from stack to output
                while (stack and 
                    stack[-1] in self.precedence and 
                    self.precedence[stack[-1]] >= self.precedence[token]):
                    output.append(stack.pop())
                stack.append(token)
                
            # Handle parentheses    
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack:
                    stack.pop()  # Remove '('

        
        # Pop remaining operators from stack to output
        while stack:
            output.append(stack.pop())
        
        return tuple(output)


    def handle_chained_comparison(self, expr):
        """Pre-process chained comparisons before converting to postfix."""
        if len(expr) >= 5:
            ops = {'<', '<=', '>', '>=', '==', '!='}
            if expr[1] in ops and expr[3] in ops:
                # Convert "a op1 b op2 c" to "(a op1 b) and (b op2 c)"
                left = expr[:3]   # a op1 b
                right = expr[2:5] # b op2 c
                return left + ['and'] + right
        return expr


    def extract_variables(self, criteria_l):
        variables = set()
        for expr in criteria_l:
            # Strip comments before extracting variables
            expr = self.strip_comments(expr)
            if expr:  # Skip empty expressions (comment-only lines)
                variables.update(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
        return list(variables)


    def create_parsing_logic(self, criteria_l):
        """Create parsing logic for a list of infix expressions.
        
        Lines starting with '#' or containing only comments are skipped.
        Inline comments (text after '#') are stripped from expressions.
        """
        # Strip comments and filter out empty/comment-only lines
        cleaned_criteria_l = []
        for expr in criteria_l:
            cleaned_expr = self.strip_comments(expr)
            if cleaned_expr:  # Skip empty expressions (comment-only lines)
                cleaned_criteria_l.append(cleaned_expr)
        
        variables = self.extract_variables(cleaned_criteria_l)
        postfix_expressions = []
        
        for expr in cleaned_criteria_l:
            tokens = re.findall(self.token_pattern, expr)
            postfix_expr = self.infix_to_postfix(tokens)
            postfix_expressions.append(postfix_expr)
        
        return variables, postfix_expressions # do not sort. leave in input order.

    def evaluate_expression_in_bt(self, eval_dfs, variables, postfix_expressions):
        """
        Evaluate parsed postfix expressions for each direction and return:
          - exprs_df: a single DataFrame containing all directional expression columns
          - exprs_dfs_d: a dict mapping each direction ('N','E','S','W') to its own DataFrame

        Args:
            eval_dfs (Dict[str, pl.DataFrame]): mapping of direction -> DataFrame with variable columns
            variables (List[str]): list of variable names extracted from criteria
            postfix_expressions (List[Tuple[str,...]]): postfix token tuples for each expression

        Returns:
            Tuple[pl.DataFrame, Dict[str, pl.DataFrame]]
        """
        # Build mapping from postfix expression -> index (zfilled to 3 in downstream)
        postfix_expressions_d = {postfix_expressions[i]: i for i in range(len(postfix_expressions))}

        # Evaluate per direction and prefix result columns with direction (e.g., 'N_expr_000')
        exprs_dfs_d = {}
        for d in 'NESW':
            if d not in eval_dfs:
                raise KeyError(f"Missing eval DataFrame for direction '{d}'")
            df = eval_dfs[d]
            exprs_dfs_d[d] = self.evaluate_expressions_with_dataframe(
                df,
                postfix_expressions_d,
                column_prefix=f"{d}_",
            )

        # Concatenate per-direction results horizontally to create a unified DataFrame
        exprs_df = pl.concat([exprs_dfs_d[d] for d in 'NESW'], how='horizontal')
        return exprs_df, exprs_dfs_d


    def evaluate_postfix_vectorized(self, postfix_expr, df):
        """Evaluate a postfix expression using vectorized operations."""
        stack = []
        for token in postfix_expr:
            if token.isnumeric():
                stack.append(np.full(len(df), int(token)))
            elif token in self.ops:
                if token == 'not':
                    if not stack:  # Check for empty stack
                        raise ValueError(f"Empty stack when processing 'not' operator. Expression: {postfix_expr}")
                    a = stack.pop()
                    result = self.ops[token](a)
                else:
                    if len(stack) < 2:  # Check for insufficient operands
                        raise ValueError(f"Insufficient operands for operator '{token}'. Stack: {stack}, Expression: {postfix_expr}")
                    b = stack.pop()
                    a = stack.pop()
                    result = self.ops[token](a, b)
                stack.append(result)
            elif re.match(r'\b[a-zA-Z_]\w*\b', token):
                if token not in df.columns:  # Check for missing column
                    raise ValueError(f"Column '{token}' not found in DataFrame. Available columns: {df.columns}")
                stack.append(df[token].to_numpy())
            else:
                raise ValueError(f"Invalid token: {token}")
                    
        if len(stack) != 1:  # Check final stack state
            raise ValueError(f"Invalid expression: {postfix_expr}. Final stack has {len(stack)} items: {stack}")
        return stack[0]


    def evaluate_expressions_with_dataframe(self, df, postfix_expressions_d, column_prefix=''):
        results_dict = {}
        
        assert len(postfix_expressions_d) <= 999, len(postfix_expressions_d) # if asserted, expand from zfill(3) to zfill(4).
        for postfix_expr,index in postfix_expressions_d.items():
            result = self.evaluate_postfix_vectorized(postfix_expr, df)
            results_dict[f'{column_prefix}expr_{str(index).zfill(3)}'] = result
        
        results = pl.DataFrame(results_dict)
        return results


    def create_eval_criteria(self, bbo_previous_bid_to_bt_entry_d):
        t = time.time()
        expressions_l = []
        for eval_expr in bbo_previous_bid_to_bt_entry_d.values():
            #print(eval_expr)
            if len(eval_expr[4]):
                expressions_l.extend(eval_expr[4])
        # Strip comments and filter out empty/comment-only expressions
        cleaned_expressions = []
        for expr in expressions_l:
            cleaned_expr = self.strip_comments(expr)
            if cleaned_expr:
                cleaned_expressions.append(cleaned_expr)
        expressions = sorted(set(cleaned_expressions))
        print(f"create_eval_criteria: done: time: {time.time()-t:.2f}") # 1s
        return expressions


    def create_directional_exprs_cols(self, criteria_variables_infix_syms):
        self.directional_exprs_cols_d = {}  # map of original expr to directional expr
        self.directional_variable_cols_d = {}  # map of base var -> directional var
        self.regex_col_names_d = {}
        # Accept either a dict (use values) or any iterable of strings
        if isinstance(criteria_variables_infix_syms, dict):
            expr_syms = list(criteria_variables_infix_syms.values())
        else:
            expr_syms = list(criteria_variables_infix_syms)
        for v in expr_syms:
            v = str(v)
            m = re.match(r'^\s*([A-Za-z][A-Za-z0-9_]*(?:_[CDHSN])?)\s*(.*)$', v)
            if not m:
                # Fallback: identity mapping
                self.directional_variable_cols_d[v] = v
                self.directional_exprs_cols_d[v] = v
                self.regex_col_names_d[v] = v
                continue
            base = m.group(1)  # variable token (may include _[CDHSN])
            tail = m.group(2)  # operator and value (may be empty)
            # Handle card pattern like 'C_CA', 'C_DQ', etc.
            if base.startswith('C_') and len(base) >= 4:
                # Insert {d} after 'C_' e.g. C_CA -> C_{d}CA
                directional_var = base.replace('C_', 'C_{d}', 1)
                regex = directional_var[:-2].replace('{d}', '([NESW])?[CDHS][AKQJT98765432]')
            else:
                parts = base.split('_')
                last = parts[-1]
                if last in ('C', 'D', 'H', 'S', 'N'):
                    # Insert {d} before the final suit/NT token: SL_C -> SL_{d}_C
                    directional_var = '_'.join(parts[:-1] + ['{d}', last])
                    regex = directional_var[:-1].replace('{d}', '([NESW]_)?[CDHS]')
                else:
                    # Non-suit variable: add direction suffix HCP -> HCP_{d}, Total_Points -> Total_Points_{d}
                    directional_var = f'{base}_{{d}}'
                    regex = directional_var.replace('{d}', '([NESW])')
            directional_expr = f'{directional_var} {tail.strip()}' if tail else directional_var
            # Populate maps: base var -> directional var; full expr -> directional expr; regex for matching columns
            self.directional_variable_cols_d[base] = directional_var
            self.directional_exprs_cols_d[v] = directional_expr
            self.regex_col_names_d[base] = regex
        # Return dicts for convenience (attributes are already set)
        return self.regex_col_names_d, self.directional_variable_cols_d, self.directional_exprs_cols_d


    def create_bidding_expr_dicts(self, bt_prior_bids_to_bt_entry_d):

        t = time.time()
        self.criteria_expressions_infix_l = self.create_eval_criteria(bt_prior_bids_to_bt_entry_d)
        print(len(self.criteria_expressions_infix_l),self.criteria_expressions_infix_l)

        # create dictionaries to map between symbol and index for postfixexpressions.
        self.criteria_variables_infix_sym_to_index_d = {v:i for i,v in enumerate(self.criteria_expressions_infix_l)}
        print(f"create_bidding_expr_dicts: infix: time: {time.time()-t:.2f}") # 1s
        print(self.criteria_variables_infix_sym_to_index_d)
        self.criteria_variables_infix_index_to_sym_d = {i:v for i,v in enumerate(self.criteria_expressions_infix_l)}
        print(f"create_bidding_expr_dicts: criteria_variables_infix_sym_to_index_d: time: {time.time()-t:.2f}") # 1s
        print(self.criteria_variables_infix_index_to_sym_d)

        # takes 10s
        # create dict of metrics (e.g. 'HCP') used by bt entry expr list. keys are metric. values are tuple of (expr, index).
        # all metrics appear as first token of every expression e.g. HCP >= 15.
        self.metric_col_d = defaultdict(list)
        for k,v in self.criteria_variables_infix_sym_to_index_d.items():
            metric = k.split()[0]
            self.metric_col_d[metric].append((k,v))
        assert all([m[0].isalpha() for m in self.metric_col_d.keys()])
        print(f"create_bidding_expr_dicts: metric_col_d: time: {time.time()-t:.2f}") # 1s
        print(len(self.metric_col_d),self.metric_col_d['HCP'])     

        # takes 0s
        # convert to directional variables
        self.create_directional_exprs_cols(self.criteria_variables_infix_index_to_sym_d.values())
        
        # takes 6-7m
        # create dict of metric column used by bt entry expr list. keys are index. values are metric.
        # all metrics appear as first token of every expression e.g. HCP >= 15.
        # preserve order of metrics in bt entry expr list. use sorted(set(bt_index_to_metric_col_d)) when needed.
        self.bt_index_to_metric_col_d = {i:[col.split()[0] for col in v[4]] for i,(k,v) in enumerate(bt_prior_bids_to_bt_entry_d.items())}
        assert all([col[0].isalpha() for l in self.bt_index_to_metric_col_d.values() for col in l])
        print(f"create_bidding_expr_dicts: bt_index_to_metric_col_d: time: {time.time()-t:.2f}") # 1s
        print(len(self.bt_index_to_metric_col_d),self.bt_index_to_metric_col_d[0],sorted(set(self.bt_index_to_metric_col_d[0])))

        # takes 30s
        # create dict of bt entry expr list. keys are index. values are expr list e.g. ['HCP >= 15', 'SL_C >= 3']
        self.bt_index_to_criteria_exprs_d = {i:v[4] for i,(k,v) in enumerate(bt_prior_bids_to_bt_entry_d.items())}
        print(f"create_bidding_expr_dicts: bt_index_to_criteria_exprs_d: time: {time.time()-t:.2f}") # 1s
        print(len(self.bt_index_to_criteria_exprs_d),self.bt_index_to_criteria_exprs_d[0])

        # takes 7m30s
        # create dict of bt entry expr list. keys are index. values are expr indexes.
        self.bt_index_to_criteria_cols_d = {i:[self.criteria_variables_infix_sym_to_index_d[e] for e in v[4]] for i,(k,v) in enumerate(bt_prior_bids_to_bt_entry_d.items())}
        print(f"bt_index_to_criteria_cols_d: bt_index_to_criteria_exprs_d: time: {time.time()-t:.2f}") # 1s
        print(len(self.bt_index_to_criteria_cols_d),self.bt_index_to_criteria_cols_d[0])

        print(f"create_bidding_expr_dicts: done: time: {time.time()-t:.2f}") # 1s

        return


class ExpressionEvaluator:
    """A class to evaluate bridge bidding expressions and create bidding tables."""
    
    def __init__(self):
        # Initialize attributes for storing expression results
        self.expr_results_d = {}
        self.next_bidding_seat_d = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
        self.incomplete_rules = set()
        
        # Initialize attributes that will be loaded from pickle files
        # These are created by CriteriaEvaluator and loaded here
        self.bt_index_to_criteria_cols_d = {}
        self.directional_exprs_cols_d = {}
        self.directional_variable_cols_d = {}
        self.criteria_expressions_infix_l = []
        self.criteria_variables_infix_sym_to_index_d = {}
        self.criteria_variables_infix_index_to_sym_d = {}


    def load_criteria_dicts(self, bt_index_to_metric_col_d, bt_index_to_criteria_exprs_d, 
                           bt_index_to_criteria_cols_d, directional_variable_cols_d, 
                           criteria_variables_infix_index_to_sym_d, directional_exprs_cols_d):
        """Load the criteria dictionaries created by CriteriaEvaluator.
        
        These dictionaries are typically loaded from the bbo_criteria_d.pkl file
        created by bbo_create_bidding_criteria_dicts.ipynb.
        
        Args:
            bt_index_to_metric_col_d: Mapping of bt index to metric columns
            bt_index_to_criteria_exprs_d: Mapping of bt index to criteria expressions  
            bt_index_to_criteria_cols_d: Mapping of bt index to criteria column indexes
            directional_variable_cols_d: Mapping of variables to directional variables
            criteria_variables_infix_index_to_sym_d: Mapping of expression index to expression string
            directional_exprs_cols_d: Mapping of full expressions to directional expressions
        """
        # Store all the loaded dictionaries directly - no reconstruction needed!
        self.bt_index_to_criteria_cols_d = bt_index_to_criteria_cols_d
        self.directional_variable_cols_d = directional_variable_cols_d
        self.criteria_variables_infix_index_to_sym_d = criteria_variables_infix_index_to_sym_d
        self.directional_exprs_cols_d = directional_exprs_cols_d
        
        # Create reverse mapping
        self.criteria_variables_infix_sym_to_index_d = {v:k for k,v in criteria_variables_infix_index_to_sym_d.items()}
        
        # Create the list of expressions in index order
        self.criteria_expressions_infix_l = [criteria_variables_infix_index_to_sym_d[i] 
                                             for i in sorted(criteria_variables_infix_index_to_sym_d.keys())]


    def create_bidding_table_boolean_columns(self, exprs_df, infix_expressions_d, bbo_bidding_table_position_d):
        assert len(bbo_bidding_table_position_d) == 4
        bt_bidding_col_results = {}
        for direction in 'NESW':
            direction = direction[0] # untuple dealer
            cached_bt0_hits = 0
            cached_binop_hits = 0
            empty_eval_exprs = 0
            binop_cache_d = {}
            pos = None  # Initialize to avoid unbound variable
            for pos,bts in bbo_bidding_table_position_d.items():
                for bt0,bt in enumerate(bts):
                    bt0 = bt[0]
                    bt_exprs = bt[4]
                    eval_expr_cols = [f'{direction}_expr_{str(infix_expressions_d[e]).zfill(3)}' for e in bt_exprs]
                    if bt0 % 100000 == 0:
                        print(direction,pos,bt0,bt)
                        print(bt_exprs)
                        print(eval_expr_cols)
                    bid = '_'.join([direction,'bt',str(bt0)])
                    if eval_expr_cols:
                        if bid in bt_bidding_col_results:
                            cached_bt0_hits += 1
                        else:
                            cache_k_ith = eval_expr_cols[0]
                            if cache_k_ith in binop_cache_d:
                                cached_binop_hits += 1
                            else:
                                # Convert to boolean type explicitly
                                binop_cache_d[cache_k_ith] = exprs_df[cache_k_ith].cast(pl.Boolean)
                            for last_row in eval_expr_cols[1:]:
                                bid_k = cache_k_ith+'&'+last_row
                                if bid_k in binop_cache_d:
                                    cached_binop_hits += 1
                                else:
                                    # Convert both operands to boolean before AND operation
                                    binop_cache_d[bid_k] = (binop_cache_d[cache_k_ith].cast(pl.Boolean) & 
                                                        exprs_df[last_row].cast(pl.Boolean))
                                cache_k_ith = bid_k
                            bt_bidding_col_results[bid] = binop_cache_d[cache_k_ith].alias(bid)
                    else:
                        assert bid not in bt_bidding_col_results, bid
                        empty_eval_exprs += 1
            print(f"dealer:{direction} pos:{pos} cached binop columns:{len(binop_cache_d)} cache binop hits:{cached_binop_hits} cached bt0 hits:{cached_bt0_hits} empty eval exprs:{empty_eval_exprs}")
            del binop_cache_d

        return bt_bidding_col_results


    def compute_bidding_table_results(self, criteria_bt_results_d, bbo_previous_bid_to_bt_entry_d):
        bt_cols_d = {}
        for direction in 'NESW':
            for k,v in tqdm.tqdm(bbo_previous_bid_to_bt_entry_d.items()):
                #print(f"{k=} {v=}")
                #if v[0] == 10000:
                #    break
                bt0 = v[0] # 0,1,...
                d_bt_bt0 = '_'.join([direction,'bt',str(bt0)]) # 'N_bt_0'
                if len(v[1]) <= 1:
                    # opening bid. no prior bids. assign current criteria results.
                    # all opening bids must be in criteria_bt_results_d -- but only when bidding table is completed (4N)
                    if d_bt_bt0 in criteria_bt_results_d:
                        bt_cols_d[d_bt_bt0] = criteria_bt_results_d[d_bt_bt0]
                    else:
                        # todo: temp code until all eval_exprs of [] are filled in with actual expressions. default is all Trues.
                        bt_cols_d[d_bt_bt0] = True 
                    continue
                prior_bt0 = bbo_previous_bid_to_bt_entry_d[(v[1][:-1],(v[1][-1],))][0] # prior result index
                prior_d_bt_bt0 = '_'.join([direction,'bt',str(prior_bt0)]) # prior result column name
                if d_bt_bt0 not in criteria_bt_results_d:
                    bt_cols_d[d_bt_bt0] = bt_cols_d[prior_d_bt_bt0] # no current criteria results. Treat as all True. ergo pass on prior results.
                elif bt_cols_d[prior_d_bt_bt0] == True:
                    bt_cols_d[d_bt_bt0] = criteria_bt_results_d[d_bt_bt0]
                else:
                    # could test for all False (sum==0) and take some resource saving shortcut.
                    bt_cols_d[d_bt_bt0] = bt_cols_d[prior_d_bt_bt0]&criteria_bt_results_d[d_bt_bt0] # AND prior result with current criteria result.
    
        return bt_cols_d


    def evaluate_expression(self, exprs, df):
        """
        Evaluate expression using asteval for all rows in the dataframe.
        Only handles single comparisons or double comparisons with a variable in the middle.
        
        Args:
            expr (str): Expression to evaluate (e.g. "10 <= HCP <= 20")
            df (polars.DataFrame): DataFrame containing columns referenced in expr
            
        Returns:
            polars.Series: defaultdict(list) with results of expression evaluation
        """

        #print(exprs)
        #if len(exprs) == 0:
        #    return [True]
        
        # Convert dataframe columns to numpy arrays for the symbol table
        t = time.time()
        value_dict = {col: df[col].to_numpy() for col in df.columns}
        
        # Create interpreter with the data in the symbol table
        aeval = Interpreter(symtable=value_dict)
        
        expr_results_d = defaultdict(list)
        for d in 'NESW':
            for expr in tqdm.tqdm(exprs):

                # Evaluate expression
                try:
                    result_array = aeval.eval(expr.replace('{d}',d))
                    if result_array is None:
                        raise ValueError(f"Expression evaluation returned None: {expr=}")
                except Exception as e:
                    #raise ValueError(f"Expression evaluation exception: {expr=} {e=}")
                    print(f"Expression evaluation exception: {expr=} {e=}")
                    continue
                
                expr_results_d[d].append(result_array)
        print(f"evaluate_expression: time: {time.time()-t:.2f}") # 1s
        return expr_results_d

    
    def create_bidding_table(self, df):

        # takes 10s.
        # Creates  len('NESW') * len(directional_exprs_cols_d) * train_df.height expressions. 4 * 349 * 1000 = 1.4m arrays.
        # Create expression dict with key of NESW, value of dict with key of index, value of expression result.
        t = time.time()
        
        # Build a list of directional expressions using the order from criteria_variables_infix_index_to_sym_d
        # This ensures we evaluate exactly the expressions that bt_index_to_criteria_cols_d will reference
        directional_exprs_list = []
        expr_indices = []
        
        # Use sorted indices to ensure consistent order
        for idx in sorted(self.criteria_variables_infix_index_to_sym_d.keys()):
            orig_expr = self.criteria_variables_infix_index_to_sym_d[idx]
            
            # Get the directional version of the expression
            if orig_expr in self.directional_exprs_cols_d:
                directional_expr = self.directional_exprs_cols_d[orig_expr]
            else:
                # No directional mapping, use original
                directional_expr = orig_expr
            
            directional_exprs_list.append(directional_expr)
            expr_indices.append(idx)
        
        print(f"About to evaluate {len(directional_exprs_list)} expressions")
        
        # Evaluate expressions (returns results in same order as input list)
        evaluate_expression_d = self.evaluate_expression(directional_exprs_list, df)
        print(f"create_bidding_table: evaluate_expression_d: time: {time.time()-t:.2f}") # 1s
        
        # Map results back to their original expression indices
        # Use the original indices, not sequential 0, 1, 2, ...
        self.expr_results_d = {}
        for d, results_list in evaluate_expression_d.items():
            self.expr_results_d[d] = {expr_indices[i]: results_list[i] for i in range(len(results_list))}
        
        print(f"create_bidding_table: done: time: {time.time()-t:.2f}") # 1s
        print(f"Stored results for {len(self.expr_results_d['N'])} expressions")
        print(len(self.expr_results_d),len(self.expr_results_d['N']),len(self.expr_results_d['N'][0]),self.expr_results_d['N'][0])

        return df


    def collect_auctions_used(self, level, d, prior_bids, row_index, bt_results_d, bt_prior_bids_to_bt_entry_d, bt_bid_to_next_bids_d):
        for bid in bt_bid_to_next_bids_d[prior_bids]:
            bt = bt_prior_bids_to_bt_entry_d[(prior_bids,bid)]
            bt0 = bt[0]
            v = self.bt_index_to_criteria_cols_d[bt0]
            #print(level,d,prior_bids,row_index,bt0,v)
            if bt0 in bt_results_d[row_index]:
                continue # already computed
            if v == []:
                #print('missing rules:',row_index,bt)
                continue
            b = self.expr_results_d[d][v[0]][row_index] # [row_index] make t a scaler boolean otherwise need .copy()
            for e in v[1:]:
                b &= self.expr_results_d[d][e][row_index] # scaler boolean
            if not b: # scaler boolean
                # False so rejecting candidate bid
                continue
            assert bt[4], (row_index,bt) # unexpected empty rules
            # todo: detecting hardcoded bidding rule errors. can be removed when bt is fixed.
            if bt[1] in [(),('p',),('p','p'),('p','p','p')] and bt[2] != ('p',) and bt[2] >= ('5H',): # todo: fix criteria for openings >= 5H
                if bt0 in self.incomplete_rules:
                    continue
                self.incomplete_rules.add(bt0)
                print('Incomplete rules detected:',row_index,bt)
                continue
            
            # Add this bid to results since it passed all criteria
            bt_results_d[row_index].add(bt0)
            
            if bt[6]: # Completed auction (ends with 3 passes)
                assert bt[1][-2:] + bt[2] == ('p','p','p'), (row_index,bt)
                #print('Final auction:',row_index,bt0,bt[1][:-2])
                continue  # Don't recurse further for completed auctions
            
            # default rule. might want to report for later vetting?
            #if bt[3].startswith('No suitable call'):
                #print('Unexpected: No suitable call:',row_index,bt)
            assert bt[1][-2:] + bt[2] != ('p','p','p') or bt[1] + bt[2] == ('p','p','p'), (row_index,bt)
            # Recurse to check possible responses to this bid
            self.collect_auctions_used(level+1, self.next_bidding_seat_d[d], bt[1]+bt[2], row_index, bt_results_d, bt_prior_bids_to_bt_entry_d, bt_bid_to_next_bids_d)
        return


    def get_auctions(self, df, bbo_prior_bids_to_bt_entry_d, bbo_bid_to_next_bids_d):
        t = time.time()
        bt_results_d = defaultdict(set)
        for row_index,d in tqdm.tqdm(enumerate(df['Dealer']),miniters=1000):
            # todo: is this loop necessary? can () be used to iterate over all inital 'p' bids? would require different handling of 'p'.
            for prior_bids in [(),('p',),('p','p'),('p','p','p')]:
                self.collect_auctions_used(0,d,prior_bids,row_index,bt_results_d,bbo_prior_bids_to_bt_entry_d,bbo_bid_to_next_bids_d)
                d = self.next_bidding_seat_d[d]
            #break
        print(f"collect_auctions_used: time: {time.time()-t:.2f}")
        return {k:sorted(v) for k,v in sorted(bt_results_d.items())}


class BiddingSequenceAnalyzer:
    """Analyzes bridge bidding sequences to track contracts, doubles, and bidding patterns."""
    
    MAX_PASSES = 3
    PASSED_OUT_INITIAL_PASSES = 4
    BID_LEVELS = '1234567'
    STRAIN_ORDER = 'CDHSN'
    
    def __init__(self):
        """Initialize analyzer."""
        self.reset()
        self.next_bidding_seat_d = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'} # todo: put into mlBridgeLib.py?
        self.pair_directions_d = {'N': 'NS', 'E': 'EW', 'S': 'NS', 'W': 'EW'} # todo: put into mlBridgeLib.py?

    def reset(self):
        """Reset all state variables to initial values."""
        # state variables
        self.npasses = 0
        self.seat_direction = None
        self.pair_direction = None
        self.last_bidder_seat = None
        self.last_bidder_pair_direction = None
        self.first_bidder_of_strain_d: Dict[Tuple[str | None, str | None], str | None] = {}

        # auction variables
        self.initial_passes = self.PASSED_OUT_INITIAL_PASSES
        self.auction_completed = False
        self.opening_bid = None
        self.opening_seat = None
        self.level = None
        self.strain = None
        self.contract = None
        self.double = False
        self.redouble = False
        self.declarer_seat = None
        self.on_lead_seat = None
    
    def is_bid_valid(self, current_bid):
        if len(current_bid) != 2 or current_bid[0] not in self.BID_LEVELS or current_bid[1] not in self.STRAIN_ORDER: # oops, must be [1-7][CDHSN]
            return False # invalid bid
        if self.contract is None: # no contract yet, no problem
            return True
        if current_bid[0] < self.contract[0]:
            return False # oops, new bid has a lower level
        if current_bid[0] == self.contract[0]:
            if self.STRAIN_ORDER.index(current_bid[1]) <= self.STRAIN_ORDER.index(self.contract[1]):
                return False # oops, new bid is same level but is a lower ranking strain
        return True
    
    def analyze_bidding_sequence(self, bids, dealer='N'):
        """Analyze a complete bidding sequence."""
        self.reset()
        assert len(bids) # empty bids is an error
        tbids = tuple(bids) # convert possible list to tuple for correct comparisons

        # initialize seat and pair
        self.seat_direction = dealer
        self.pair_direction = self.pair_directions_d[self.seat_direction]

        for bid_number,current_bid in enumerate(tbids):
        
            #print(f"{bid_number=} {current_bid=} {self.seat_direction=} {self.pair_direction=} {tbids=}")
            match current_bid:
                # Invalid if more than 3 passes unless it's a passed out hand
                case 'p':
                    self.npasses += 1
                    # Invalid if more than 3 passes unless it's a passed out hand
                    if self.npasses > self.MAX_PASSES and tbids != ('p','p','p','p'):
                        return True  # Error occurred
                        
                case 'd':
                    # Invalid if no contract, already doubled/redoubled, or too many passes
                    if (not self.contract or 
                        self.double or 
                        self.redouble or 
                        self.npasses >= self.MAX_PASSES):
                        return True  # Error occurred
                    
                    self.double = True
                    self.npasses = 0
                    
                case 'r':
                    # Invalid if no contract/seat so far, not doubled, already redoubled, or too many passes
                    if (not self.contract or 
                        self.seat_direction is None or 
                        not self.double or 
                        self.redouble or 
                        self.npasses >= self.MAX_PASSES):
                        return True  # Error occurred
                        
                    self.double = False
                    self.redouble = True
                    self.npasses = 0
                    
                case _:
                    # invalid if more than 3 passes or new bid is invalid
                    if self.npasses > self.MAX_PASSES or not self.is_bid_valid(current_bid):
                        return True  # Error occurred
                        
                    self.contract = current_bid
                    self.double = False
                    self.redouble = False
                    
                    if self.opening_bid is None:
                        self.opening_bid = self.contract
                        self.opening_seat = self.seat_direction
                        self.initial_passes = self.npasses
                    
                    self.level = self.contract[0]
                    self.strain = self.contract[1]
                    if (self.pair_direction, self.strain) not in self.first_bidder_of_strain_d:
                        self.first_bidder_of_strain_d[(self.pair_direction, self.strain)] = self.seat_direction
                    self.last_bidder_seat = self.seat_direction
                    self.last_bidder_pair_direction = self.pair_direction
                    self.npasses = 0

            # increment to next seat and pair directions unless sequence is completed
            self.seat_direction = self.next_bidding_seat_d[self.seat_direction]
            self.pair_direction = self.pair_directions_d[self.seat_direction]

        # auction completed
        if len(tbids) > 3 and tbids[-3:] == ('p','p','p'):
            self.auction_completed = True

        # although auction may not be completed, determine declarer and on lead as if auction was completed.
        if self.last_bidder_pair_direction and self.strain:
            self.declarer_seat = self.first_bidder_of_strain_d[(self.last_bidder_pair_direction, self.strain)]
            if self.declarer_seat is not None:
                self.on_lead_seat = self.next_bidding_seat_d[self.declarer_seat]

        return False  # No error


class AuctionFinder:
    def __init__(self,bt_prior_bids_to_bt_entry_d,bt_bid_to_next_bids_d,exprStr_to_exprID_d):
        # kludge to work around bogus bidding sequences
        self.bogus_bidding_sequences = {
            ('4N',): True,
            ('p','4N',): True,
            ('p','p','4N',): True,
            ('p','p','p','4N',): True,
            ('5H',): True,
            ('p','5H',): True,
            ('p','p','5H',): True,
            ('p','p','p','5H',): True,
            ('5S',): True,
            ('p','5S',): True,
            ('p','p','5S',): True,
            ('p','p','p','5S',): True,
            ('5N',): True,
            ('p','5N',): True,
            ('p','p','5N',): True,
            ('p','p','p','5N',): True,
        }
        self.bt_bid_to_next_bids_d = bt_bid_to_next_bids_d
        self.bt_prior_bids_to_bt_entry_d = bt_prior_bids_to_bt_entry_d
        self.next_bidding_seat_d = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'} # todo: put into mlBridgeLib.py?
        self.exprStr_to_exprID_d = exprStr_to_exprID_d

    def auction_finder(self, level, dealer, bidding_seat, prior_bids, r, auction_d, completed_auctions_d):
        valid_sequence_found = 0
        next_bids = self.bt_bid_to_next_bids_d.get(prior_bids, [])
        for candidate_bid in next_bids:
            new_prior_bids = prior_bids + candidate_bid
            # todo: kludge -- must eliminate bogus bidding sequences until bidding table entries are cleaned and vetted.
            if new_prior_bids in self.bogus_bidding_sequences:
                continue
            # check that (prior_bids, candidate_bid) are in bidding table. if not, default of pass will be used.
            if (prior_bids, candidate_bid) in self.bt_prior_bids_to_bt_entry_d:
                #print(f"{prior_bids=} {candidate_bid=} {self.bt_prior_bids_to_bt_entry_d[(prior_bids, candidate_bid)]=}")
                # (prior_bids, candidate_bid) are in bidding table.
                bt = self.bt_prior_bids_to_bt_entry_d[(prior_bids, candidate_bid)]
                bt_col = '_'.join([bidding_seat, 'bt', str(bt[0])])
                #print(f"{bt_col=}")
                # if bt_col is not in r, then assume the column was eliminated because it had all False values and continue.
                if bt_col in r:
                    #print(f"{bt_col} in r")
                    # column exists. use the bid only if it meets bid's criteria (True).
                    if r[bt_col]:
                        #print(f"{bt_col} is True")
                        # bid meets its criteria. add to auction.
                        assert new_prior_bids not in auction_d, (new_prior_bids, level, bidding_seat, prior_bids)
                        auction_d[new_prior_bids] = bidding_seat
                        valid_sequence_found += 1
                        next_bidding_seat = self.next_bidding_seat_d[bidding_seat]
                        self.auction_finder(level + 1, dealer, next_bidding_seat, new_prior_bids, r, auction_d, completed_auctions_d)
        #     else:
        #         # candidate bid is not in bidding table. use pass as the default bid and continue the search for candidate bids unless passed out.
        #         # hmmm, this is where not having every intermediate bid in the bidding table is confusing.
        #         if len(prior_bids) < 4 or (prior_bids[-3:] != ('p', 'p', 'p') and prior_bids != ('p', 'p', 'p')):
        #             new_prior_bids = prior_bids + ('p',)
        #             assert new_prior_bids not in auction_d, (new_prior_bids, level, bidding_seat, prior_bids)
        #             auction_d[new_prior_bids] = bidding_seat
        #             valid_sequence_found += 1
        #             next_bidding_seat = self.next_bidding_seat_d[bidding_seat]
        #             self.auction_finder(level + 1, dealer, next_bidding_seat, new_prior_bids, r, auction_d, completed_auctions_d)
        if valid_sequence_found == 0:
            # if no valid sequences were found, then use pass as the default bid and continue the search for candidate bids unless passed out..
            new_prior_bids = prior_bids + ('p',)
            #print(f"{prior_bids=} {len(prior_bids)=} {prior_bids[-3:]=} {prior_bids=}")
            if len(prior_bids) < 4 or (prior_bids[-3:] != ('p', 'p', 'p') and prior_bids != ('p', 'p', 'p')):
                auction_d[new_prior_bids] = bidding_seat
                next_bidding_seat = self.next_bidding_seat_d[bidding_seat]
                self.auction_finder(level + 1, dealer, next_bidding_seat, new_prior_bids, r, auction_d, completed_auctions_d)
            else:
                # full stop. adding a pass has resulted in a passed out auction. no way to continue.
                # add the passed out auction to completed_auctions_d only if there's just no other auction available.
                # todo: completed auctionstuff is now directly available in bbo_prior_bids_to_bt_entry_d
                if len(completed_auctions_d) == 0 or completed_auctions_d != ('p', 'p', 'p', 'p'):
                    assert new_prior_bids not in completed_auctions_d, (new_prior_bids, level, bidding_seat, prior_bids)
                    completed_auctions_d[new_prior_bids[:-3]] = dealer
        return auction_d, completed_auctions_d
    
    
    def augment_df_with_bidding_info(self, df):
        # create a df with ai bidding sequences
    
        analyser = BiddingSequenceAnalyzer()

        exprs_dfs_d = {}
        elapsed_time = 0
        for r in df.iter_rows(named=True):
            #print(f"{r['PBN']}")
            dealer = r['Dealer']
            bidding_seat = r['Dealer']
            prior_bids = ()
            t = time.time()
            auction_d, completed_auctions_d = self.auction_finder(0,dealer,bidding_seat,prior_bids,r,{},{})
            elapsed_time += time.time()-t
            print(f"augment_df_with_bidding_info: {dealer=} {bidding_seat=} {prior_bids=} auction_d:{auction_d=} completed_auctions_d:{completed_auctions_d=}")
            expr_d = defaultdict(list)
    
            # todo: is auction_d really needed or can we just use completed_auctions_d?
            # create rows for each bid in auction. for observing progression of auction sequence.
            for new_prior_bids,seat in auction_d.items():
                prior_bids = new_prior_bids[:-1]
                candidate_bid = (new_prior_bids[-1],)
                
                # call analyze_bidding_sequence to validate auction
                if analyser.analyze_bidding_sequence(new_prior_bids, dealer=dealer):
                    raise ValueError(f"Invalid auction: {new_prior_bids}") # this should never happen because auction was validated in previous step.
                opening_seat = analyser.opening_seat
                declarer = analyser.declarer_seat
                on_lead_seat = analyser.on_lead_seat
                auction_completed = analyser.auction_completed

                # hand record
                expr_d['index'].append(r['index'])
                expr_d['Board'].append(r['Board'])
                expr_d['PBN'].append(r['PBN'])
                expr_d['Dealer'].append(r['Dealer'])
                expr_d['Vul'].append(r['Vul'])

                # bidding
                expr_d['Opening_Seat'].append(opening_seat)
                expr_d['Auction_Completed'].append(auction_completed)
                expr_d['Declarer'].append(declarer)
                expr_d['OnLead_Seat'].append(on_lead_seat)
                expr_d['prior_bids'].append('-'.join(b for b in prior_bids) if prior_bids else 'None') # converting list of string to string. could use list(bt[1])
                expr_d['candidate_bid'].append(candidate_bid)
                #print(f"{prior_bids=} {candidate_bid=} {(prior_bids,candidate_bid) in self.bt_prior_bids_to_bt_entry_d=}")
                #if (prior_bids,candidate_bid) in self.bt_prior_bids_to_bt_entry_d:
                #    print(f"{'_'.join([seat,'bt',str(self.bt_prior_bids_to_bt_entry_d[(prior_bids,candidate_bid)][0])])=}")
                # be sure to keep both THEN and ELSE in sync with column names and dtypes.
                if (prior_bids,candidate_bid) in self.bt_prior_bids_to_bt_entry_d and '_'.join([seat,'bt',str(self.bt_prior_bids_to_bt_entry_d[(prior_bids,candidate_bid)][0])]) in r:
                    bt = self.bt_prior_bids_to_bt_entry_d[(prior_bids,candidate_bid)]
                    bt_col = '_'.join([seat,'bt',str(bt[0])])
                    expr_d['bt_index'].append(bt_col)
                    expr_d['candidate_accepted'].append(r[bt_col] if r[bt_col] else True) # True if empty?
                    expr_d['candidate_desc'].append(bt[3])
                    criteria_str = bt[4]
                    expr_d['criteria_str'].append(criteria_str)
                    criteria_col = [f"{seat}_expr_{str(self.exprStr_to_exprID_d[es]).zfill(3)}" for es in criteria_str]
                    expr_d['criteria_col'].append(criteria_col)
                    criteria_values = [r[cid] for cid in criteria_col]
                    expr_d['criteria_values'].append(criteria_values)
                else:
                    expr_d['bt_index'].append(None)
                    expr_d['candidate_accepted'].append(True)
                    expr_d['candidate_desc'].append('None')
                    expr_d['criteria_str'].append(['None'])
                    expr_d['criteria_col'].append(['None'])
                    expr_d['criteria_values'].append([True])
            expr_df = pl.DataFrame(expr_d)
            assert r['index'] not in exprs_dfs_d, r['index']
            exprs_dfs_d[r['index']] = expr_df
        print(f"augment_df_with_bidding_info took {elapsed_time:.2f}s to bid {len(df)} auctions. avg of {elapsed_time/len(df):.2f}s per auction")
        return exprs_dfs_d


# =============================================================================
# Bidding Query Functions (moved from bbo_bidding_queries_lowmem.py)
# =============================================================================

# ---------------------------------------------------------------------------
# Paths & constants for bidding queries
# ---------------------------------------------------------------------------

DIRECTIONS = ["N", "E", "S", "W"]
KEYWORDS = {"and", "or", "not", "True", "False"}
BIDDING_KEYWORDS = {"and", "or", "not", "True", "False"}  # Alias for backward compatibility


# ---------------------------------------------------------------------------
# Helpers for loading execution-plan metadata
# ---------------------------------------------------------------------------

def load_execution_plan_data(
    exec_plan_file: pathlib.Path,
) -> Tuple[List[str], Dict[str, Dict[str, str]], List[str], Dict[str, Dict[str, str]]]:
    """Load pre-computed execution-plan data from pickle."""
    t0 = time.time()
    with open(exec_plan_file, "rb") as f:
        saved_data: Dict[str, Any] = pickle.load(f)

    directionless_criteria_cols: List[str] = saved_data["directionless_criteria_cols"]
    expr_map_by_direction = saved_data["expr_map_by_direction"]
    valid_deal_columns = saved_data["valid_deal_columns"]
    pythonized_exprs_by_direction = saved_data["pythonized_exprs_by_direction"]
    del saved_data

    print("Loaded pre-computed data (execution plan)")
    print(f"  directionless_criteria_cols: {len(directionless_criteria_cols)} items")
    print(f"  expr_map_by_direction: {len(expr_map_by_direction)} directions")
    print(f"  valid_deal_columns: {len(valid_deal_columns)} columns")
    print(f"  pythonized_exprs_by_direction: {len(pythonized_exprs_by_direction)} directions")
    print(f"  load_execution_plan_data: {time.time() - t0:.2f}s")

    return directionless_criteria_cols, expr_map_by_direction, valid_deal_columns, pythonized_exprs_by_direction


# ---------------------------------------------------------------------------
# Low-memory loaders for the large DataFrames
# ---------------------------------------------------------------------------

def load_deal_df(
    bbo_mldf_augmented_file: pathlib.Path,
    valid_deal_columns: List[str],
    mldf_n_rows: int | None = None,
) -> pl.DataFrame:
    """Load `deal_df` with memory-saving tricks."""

    display_cols = ["index", "Hand_N", "Hand_E", "Hand_S", "Hand_W", "Dealer"]
    columns_to_load = sorted(set(valid_deal_columns).union(display_cols))

    print(f"Loading {len(columns_to_load)} columns into deal_df (criteria + display)...")
    t0 = time.time()
    deal_df = pl.read_parquet(
        bbo_mldf_augmented_file,
        columns=columns_to_load,
        n_rows=mldf_n_rows,
    )
    print(f"Loaded deal_df: shape={deal_df.shape} in {time.time() - t0:.2f}s")

    # Memory note:
    # - Dealer has only 4 values → categorical is a clear win.
    # - Hand_* columns are (nearly) unique per row → categorical usually *increases* memory
    #   (dictionary of unique strings + codes), so don't cast those.
    if "Dealer" in deal_df.columns:
        deal_df = deal_df.with_columns(pl.col("Dealer").cast(pl.Categorical))
        print("Converted deal_df['Dealer'] to Categorical")

    # --- DOWNCASTING FIX ---
    # Automatically downcast numeric columns to UInt8/Int8/Int16 where safe
    print("Downcasting deal_df columns...")
    cast_ops = []
    for c in deal_df.columns:
        dtype = deal_df.schema[c]
        if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32, pl.Float64, pl.Float32):
            # Check column names for bridge-specific patterns
            col_expr = pl.col(c)
            if dtype in (pl.Float64, pl.Float32):
                col_expr = col_expr.fill_nan(None)

            if any(x in c for x in ["HCP", "SL_", "Total_Points", "Tricks", "Result"]):
                # These are always small (0-100ish)
                cast_ops.append(col_expr.fill_null(0).cast(pl.UInt8 if "UInt" in str(dtype) or "Float" in str(dtype) else pl.Int8))
            elif any(x in c for x in ["Score", "ParScore", "DD_Score", "EV_"]):
                # These can be larger or have decimals (EV)
                if "EV_" in c or dtype in (pl.Float64, pl.Float32):
                    cast_ops.append(col_expr.cast(pl.Float32))
                else:
                    cast_ops.append(col_expr.fill_null(0).cast(pl.Int16))
    
    if cast_ops:
        deal_df = deal_df.with_columns(cast_ops)
    # -----------------------

    return deal_df


def load_bt_df(
    bbo_bidding_table_augmented_file: pathlib.Path,
    include_expr_and_sequences: bool = False,
) -> pl.DataFrame:
    """Load `bt_df` with memory-saving tricks."""

    # add "Announcement" column after 'Auction' column?
    base_cols = [
        "is_opening_bid", "seat", "Auction",
        "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4",
    ]
    extra_cols = ["Expr", "is_completed_auction", "previous_bid_indices"] if include_expr_and_sequences else []
    cols_to_load = base_cols + extra_cols

    print(f"Loading bt_df columns: {cols_to_load} (include_expr_and_sequences={include_expr_and_sequences})")
    t0 = time.time()
    bt_df = pl.read_parquet(bbo_bidding_table_augmented_file, columns=cols_to_load).with_row_index("index")
    print(f"Loaded bt_df: shape={bt_df.shape} in {time.time() - t0:.2f}s")

    assert bt_df['seat'].dtype == pl.UInt8
    assert bt_df['is_opening_bid'].dtype == pl.Boolean
    assert bt_df['is_completed_auction'].dtype == pl.Boolean
    #bool_cols = [c for c in ["is_opening_bid", "is_completed_auction"] if c in bt_df.columns]
    #if bool_cols:
    #    bt_df = bt_df.with_columns([pl.col(c).cast(pl.Boolean) for c in bool_cols])
    # todo: is this necessary? try removing it and see if it works.
    if "Auction" in bt_df.columns:
        bt_df = bt_df.with_columns(pl.col("Auction").cast(pl.Categorical))
        print("Converted bt_df['Auction'] to Categorical")

    return bt_df


# ---------------------------------------------------------------------------
# Criteria DataFrame helpers
# ---------------------------------------------------------------------------

def directional_to_directionless(
    criteria_deal_dfs_directional: Dict[str, pl.DataFrame],
    expr_map_by_direction: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, pl.DataFrame], Dict[int, Dict[str, pl.DataFrame]]]:
    """Convert directional criteria back to directionless and organize by seat."""
    deal_criteria_by_direction_dfs: Dict[str, pl.DataFrame] = {}
    for d in DIRECTIONS:
        dir_df = criteria_deal_dfs_directional[d]
        mapping = expr_map_by_direction[d]
        inv_mapping = {v: k for k, v in mapping.items()}
        deal_criteria_by_direction_dfs[d] = dir_df.rename(inv_mapping)

    deal_criteria_by_seat_dfs: Dict[int, Dict[str, pl.DataFrame]] = {seat: {} for seat in range(1, 5)}
    for dealer in DIRECTIONS:
        dealer_idx = DIRECTIONS.index(dealer)
        for seat in range(1, 5):
            direction = DIRECTIONS[(dealer_idx + seat - 1) % 4]
            deal_criteria_by_seat_dfs[seat][dealer] = deal_criteria_by_direction_dfs[direction]

    return deal_criteria_by_direction_dfs, deal_criteria_by_seat_dfs


def create_directional_deal_criteria_dfs(
    df: pl.DataFrame,
    pythonized_exprs_by_direction: Dict[str, Dict[str, str]],
    expr_map_by_direction: Dict[str, Dict[str, str]],
) -> Dict[str, pl.DataFrame]:
    """Create criteria DataFrames using pre-pythonized expressions.

    This is adapted from the notebook version but unchanged in behavior.
    """

    eval_env = {col: pl.col(col) for col in df.columns}

    def expr_to_polars(pythonized_expr: str) -> pl.Expr:
        # Extract tokens to check for missing columns
        tokens = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", pythonized_expr)
        tokens = [t for t in tokens if t not in KEYWORDS and t != "pl" and t != "lit"]
        missing = [t for t in tokens if t not in eval_env]
        if missing:
            # If columns are missing, treat expression as always true
            # todo: need to fix missing columns. return a column with the missing column name and a boolean value of True.
            print(f"Important: Missing columns found. Defaulting to False. Need to fix this in previous steps: {missing}")
            return pl.lit(False)
        s = re.sub(r"\bTrue\b", "pl.lit(True)", pythonized_expr)
        s = re.sub(r"\bFalse\b", "pl.lit(False)", s)
        return eval(s, {"pl": pl}, eval_env)

    criteria_dfs_directional: Dict[str, pl.DataFrame] = {}
    for d in DIRECTIONS:
        expr_objects = []
        for orig_expr, pythonized_expr in pythonized_exprs_by_direction[d].items():
            directional_name = expr_map_by_direction[d][orig_expr]
            expr_obj = expr_to_polars(pythonized_expr)
            if not isinstance(expr_obj, pl.Expr):
                expr_obj = pl.lit(bool(expr_obj))
            expr_objects.append(expr_obj.alias(directional_name))
        criteria_dfs_directional[d] = df.select(expr_objects)
    return criteria_dfs_directional


def _get_bitmap_path(deal_file: pathlib.Path) -> pathlib.Path:
    """Get single bitmap file path based on the deal file name.
    
    For deal file 'bbo_mldf_augmented.parquet', bitmap is named:
    - bbo_mldf_augmented_criteria_bitmaps.parquet
    
    The file contains columns for all 4 directions with prefixes: DIR_N_, DIR_E_, DIR_S_, DIR_W_
    """
    stem = deal_file.stem  # e.g., 'bbo_mldf_augmented'
    parent = deal_file.parent
    return parent / f"{stem}_criteria_bitmaps.parquet"


def _bitmap_file_is_stale(
    deal_file: pathlib.Path,
    exec_plan_file: pathlib.Path,
    bitmap_path: pathlib.Path,
) -> Tuple[bool, str]:
    """Check if bitmap file needs rebuilding.
    
    Returns (is_stale, reason) tuple.
    """
    # Check if bitmap file is missing
    if not bitmap_path.exists():
        return True, f"bitmap file missing: {bitmap_path.name}"
    
    # Check if source files are newer than bitmap
    source_files = [deal_file, exec_plan_file]
    missing_sources = [f for f in source_files if not f.exists()]
    if missing_sources:
        return True, f"source files missing: {[str(f) for f in missing_sources]}"
    
    # Get bitmap mtime and newest source mtime
    bitmap_mtime = bitmap_path.stat().st_mtime
    newest_source_mtime = max(f.stat().st_mtime for f in source_files)
    
    if newest_source_mtime > bitmap_mtime:
        # Find which source is newer
        for f in source_files:
            if f.stat().st_mtime > bitmap_mtime:
                return True, f"source file newer than bitmap: {f.name}"
    
    return False, "bitmap is fresh"


def _combine_direction_dfs(dfs_by_direction: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Combine 4 direction DataFrames into one with prefixed column names."""
    combined_cols = []
    for direction in DIRECTIONS:
        df = dfs_by_direction[direction]
        # Prefix each column with DIR_{direction}_
        for col in df.columns:
            combined_cols.append(df[col].alias(f"DIR_{direction}_{col}"))
    return pl.DataFrame().with_columns(combined_cols) if combined_cols else pl.DataFrame()


def _split_combined_df(combined_df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Split a combined DataFrame back into 4 direction DataFrames."""
    result = {}
    for direction in DIRECTIONS:
        prefix = f"DIR_{direction}_"
        # Find columns for this direction and strip prefix
        dir_cols = [c for c in combined_df.columns if c.startswith(prefix)]
        if dir_cols:
            # Select and rename columns (strip prefix)
            renamed = [pl.col(c).alias(c[len(prefix):]) for c in dir_cols]
            result[direction] = combined_df.select(renamed)
        else:
            result[direction] = pl.DataFrame()
    return result


def build_or_load_directional_criteria_bitmaps(
    deal_df: pl.DataFrame,
    pythonized_exprs_by_direction: Dict[str, Dict[str, str]],
    expr_map_by_direction: Dict[str, Dict[str, str]],
    deal_file: pathlib.Path | None = None,
    exec_plan_file: pathlib.Path | None = None,
) -> Dict[str, pl.DataFrame]:
    """Build or load per-direction criteria DataFrames.

    Each returned DataFrame for a direction contains one boolean column per
    criterion, with as many rows as deals.
    
    If deal_file and exec_plan_file are provided, attempts to load cached
    bitmap file. If cached file is missing or stale (source files are newer),
    rebuilds and saves the bitmap.
    
    Bitmap file is named based on the deal file:
    - {deal_file_stem}_criteria_bitmaps.parquet
    
    The file contains all 4 directions with column prefixes: DIR_N_, DIR_E_, etc.
    
    Args:
        deal_df: DataFrame of deals to evaluate criteria against
        pythonized_exprs_by_direction: Criteria expressions per direction
        expr_map_by_direction: Maps original expressions to directional names
        deal_file: Path to the deal parquet file (for cache naming/staleness check)
        exec_plan_file: Path to execution plan pickle (for staleness check)
    
    Returns:
        Dict mapping direction (N/E/S/W) to DataFrame of boolean criteria columns
    """
    # If no file paths provided, always build (legacy behavior)
    if deal_file is None or exec_plan_file is None:
        print("[bitmaps] No file paths provided, building fresh (no caching)")
        t0 = time.time()
        criteria_deal_dfs_directional = create_directional_deal_criteria_dfs(
            deal_df, pythonized_exprs_by_direction, expr_map_by_direction
        )
        print(f"[bitmaps] Built in {time.time() - t0:.1f}s")
        return criteria_deal_dfs_directional
    
    # Get bitmap file path
    bitmap_path = _get_bitmap_path(deal_file)
    
    # Check if bitmap is stale
    is_stale, reason = _bitmap_file_is_stale(deal_file, exec_plan_file, bitmap_path)
    
    if not is_stale:
        # Load from cache
        print(f"[bitmaps] Loading cached bitmap ({reason})")
        t0 = time.time()
        combined_df = pl.read_parquet(bitmap_path)
        result = _split_combined_df(combined_df)
        total_cols = sum(len(df.columns) for df in result.values())
        print(f"[bitmaps] Loaded {combined_df.height:,} rows, {total_cols} cols from {bitmap_path.name} in {time.time() - t0:.1f}s")
        for direction, df in result.items():
            print(f"  {direction}: {len(df.columns)} cols")
        return result
    
    # Build fresh
    print(f"[bitmaps] Building fresh ({reason})")
    t0 = time.time()
    criteria_deal_dfs_directional = create_directional_deal_criteria_dfs(
        deal_df, pythonized_exprs_by_direction, expr_map_by_direction
    )
    build_time = time.time() - t0
    print(f"[bitmaps] Built in {build_time:.1f}s")
    
    # Combine and save to cache
    print("[bitmaps] Saving to cache...")
    t0 = time.time()
    combined_df = _combine_direction_dfs(criteria_deal_dfs_directional)
    combined_df.write_parquet(bitmap_path)
    size_mb = bitmap_path.stat().st_size / (1024 * 1024)
    total_cols = sum(len(df.columns) for df in criteria_deal_dfs_directional.values())
    print(f"[bitmaps] Saved {combined_df.height:,} rows, {total_cols} cols → {bitmap_path.name} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")
    
    return criteria_deal_dfs_directional


def get_direction_for_seat(dealer: str, seat: int) -> str:
    dealer_idx = DIRECTIONS.index(dealer)
    return DIRECTIONS[(dealer_idx + seat - 1) % 4]


# ---------------------------------------------------------------------------
# Opening-bid search
# ---------------------------------------------------------------------------

def find_all_opening_bids_by_seat(
    dealer_deal_criteria: pl.DataFrame,
    bt_openings_df: pl.DataFrame,
    seat: int,
) -> pl.DataFrame:
    """Find opening bids for a single seat using vectorized operations."""
    n_deals = dealer_deal_criteria.height
    req_col = f"Agg_Expr_Seat_{seat}"
    empty_result = lambda: pl.DataFrame(
        {"candidate_bids": [[] for _ in range(n_deals)]},
        schema={"candidate_bids": pl.List(pl.UInt32)},
    )

    if bt_openings_df.height == 0:
        return empty_result()

    grouped = bt_openings_df.group_by(req_col).agg(pl.col("index").alias("indices"))
    exprs = []
    meta = []
    available_cols = set(dealer_deal_criteria.columns)

    for i, row in enumerate(grouped.iter_rows(named=True)):
        reqs = row[req_col]
        indices = row["indices"]
        if not reqs:
            e = pl.lit(True)
        else:
            valid = [c for c in reqs if c in available_cols]
            if len(valid) < len(reqs):
                continue
            e = pl.all_horizontal(valid)
        exprs.append(e.alias(f"m_{i}"))
        meta.append(indices)

    if not exprs:
        return empty_result()

    mask_df = dealer_deal_criteria.select(exprs)
    del grouped, exprs

    matches = []
    for i, col_name in enumerate(mask_df.columns):
        if mask_df[col_name].any():
            matched_rows = mask_df[col_name].arg_true().cast(pl.UInt32)
            matches.append((matched_rows, meta[i]))

    if not matches:
        del mask_df, meta
        return empty_result()

    del mask_df

    temp = pl.concat([
        pl.DataFrame({
            "row_id": rows,
            "bids": pl.repeat(idxs, len(rows), dtype=pl.List(pl.UInt32), eager=True),
        })
        for rows, idxs in matches
    ])

    grouped_result = temp.explode("bids").group_by("row_id").agg(pl.col("bids"))
    skeleton = pl.DataFrame({"row_id": pl.int_range(0, n_deals, dtype=pl.UInt32, eager=True)})
    final = skeleton.join(grouped_result, on="row_id", how="left")
    del matches, meta, temp, grouped_result, skeleton

    return final.select(pl.col("bids").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))).alias("candidate_bids"))


def process_opening_bids(
    deal_df: pl.DataFrame,
    bt_df: pl.DataFrame,
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, pl.DataFrame]],
    bt_parquet_file: pathlib.Path,
    seats: List[int] | None = None,
    directions: List[str] | None = None,
    opening_directions: List[str] | None = None,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Find opening bids by dealer and seat."""
    elapsed_time = time.time()
    results: Dict[Tuple[str, int], Dict[str, Any]] = {}

    seats_to_process = seats if seats is not None else [1, 2, 3, 4]
    directions_to_process = directions if directions is not None else DIRECTIONS

    valid_combos: set[Tuple[str, int]] = set()
    if opening_directions is not None:
        for dealer in directions_to_process:
            dealer_idx = DIRECTIONS.index(dealer)
            for seat in seats_to_process:
                opener = DIRECTIONS[(dealer_idx + seat - 1) % 4]
                if opener in opening_directions:
                    valid_combos.add((dealer, seat))
    else:
        for dealer in directions_to_process:
            for seat in seats_to_process:
                valid_combos.add((dealer, seat))

    print(f"Processing {len(valid_combos)} (dealer, seat) combinations...")
    openings_by_seat: Dict[int, pl.DataFrame] = {}

    for dealer in directions_to_process:
        dealer_mask = deal_df["Dealer"] == dealer
        if not dealer_mask.any():
            print(f"Skipping Dealer={dealer} (no deals in dataset)")
            continue

        dealer_indices = dealer_mask.arg_true()

        for seat in seats_to_process:
            if (dealer, seat) not in valid_combos:
                continue

            agg_col = f"Agg_Expr_Seat_{seat}"

            if seat not in openings_by_seat:
                if agg_col not in bt_df.columns:
                    print(f"Loading {agg_col} from parquet for seat {seat}")
                    agg_expr_col = pl.read_parquet(bt_parquet_file, columns=[agg_col])
                    bt_df_with_agg = bt_df.with_columns(agg_expr_col[agg_col])
                    del agg_expr_col
                else:
                    bt_df_with_agg = bt_df

                openings_by_seat[seat] = bt_df_with_agg.filter(
                    (pl.col("seat") == seat)
                    & pl.col("is_opening_bid")
                    & pl.col(agg_col).list.len().gt(1)
                )
                if bt_df_with_agg is not bt_df:
                    del bt_df_with_agg

            bt_openings_df = openings_by_seat[seat]
            criteria = deal_criteria_by_seat_dfs[seat][dealer][dealer_indices]

            t = time.time()
            new_candidates = find_all_opening_bids_by_seat(criteria, bt_openings_df, seat)
            elapsed = max(time.time() - t, 1e-6)

            direction = get_direction_for_seat(dealer, seat)
            print(f"Dealer={dealer}, Seat {seat} ({direction}): {elapsed:.2f}s, "
                  f"{criteria.height / elapsed:.0f}/sec, shape={new_candidates.shape}")

            results[(dealer, seat)] = {
                "candidates": new_candidates,
                "original_indices": dealer_indices,
            }

            del criteria
            gc.collect()

    total_elapsed = time.time() - elapsed_time
    print(f"\nTotal result keys: {len(results)} in {total_elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Opening-bid search (seat1-only BT architecture)
# ---------------------------------------------------------------------------

def build_opening_bids_table_from_bt_seat1(bt_seat1_df: pl.DataFrame) -> pl.DataFrame:
    """Build a tiny, seat-addressable opening-bids table from `bt_seat1_df`.
    
    Why this exists:
    - The runtime API should not load `bbo_bt_augmented.parquet`.
    - `bt_seat1_df` contains the *opening-call* rows (e.g. '1C', '1D', ...) with
      the opener's requirements in `Agg_Expr_Seat_1`.
    - For seats 2-4 we synthesize equivalent opening-call rows by *relabeling*
      the requirements into `Agg_Expr_Seat_{seat}` and attaching a `seat` column.
    
    Output columns (minimal set used by the API + match engine):
    - index: UInt32 synthetic primary key = bt_index * 4 + (seat-1)
    - bt_index: UInt32 original bt_index from bt_seat1_df (for debugging/reference)
    - seat: UInt8 1..4 (the opening seat being modeled)
    - Auction: Utf8 opening call (no leading p-)
    - Expr: optional expression list/struct passthrough
    - Agg_Expr_Seat_1..4: lists, with exactly one populated (the seat being modeled)
    """
    required_cols = {"bt_index", "Auction", "is_opening_bid", "Agg_Expr_Seat_1"}
    missing = sorted(required_cols - set(bt_seat1_df.columns))
    if missing:
        raise ValueError(f"bt_seat1_df missing required columns for opening-bid build: {missing}")

    base = (
        bt_seat1_df
        .filter(pl.col("is_opening_bid") & pl.col("Agg_Expr_Seat_1").list.len().gt(1))
        .select([c for c in ["bt_index", "Auction", "Expr", "Agg_Expr_Seat_1"] if c in bt_seat1_df.columns])
        .with_columns(
            pl.col("bt_index").cast(pl.UInt32).alias("bt_index"),
            # Opening-bid rows should not need a leading pass prefix. Strip defensively.
            pl.col("Auction").cast(pl.Utf8).str.replace(r"^(?:p-)+", "").alias("Auction"),
        )
    )

    # Empty list literal of the correct dtype for Agg_Expr columns.
    empty_req_list = pl.lit([], dtype=pl.List(pl.Utf8))

    per_seat: list[pl.DataFrame] = []
    for seat in (1, 2, 3, 4):
        per_seat.append(
            base.with_columns(
                (pl.col("bt_index") * pl.lit(4, dtype=pl.UInt32) + pl.lit(seat - 1, dtype=pl.UInt32)).alias("index"),
                pl.lit(seat).cast(pl.UInt8).alias("seat"),
                # Populate only the requested seat's requirements; others are empty.
                (pl.col("Agg_Expr_Seat_1") if seat == 1 else empty_req_list).alias("Agg_Expr_Seat_1"),
                (pl.col("Agg_Expr_Seat_1") if seat == 2 else empty_req_list).alias("Agg_Expr_Seat_2"),
                (pl.col("Agg_Expr_Seat_1") if seat == 3 else empty_req_list).alias("Agg_Expr_Seat_3"),
                (pl.col("Agg_Expr_Seat_1") if seat == 4 else empty_req_list).alias("Agg_Expr_Seat_4"),
            )
        )

    out = pl.concat(per_seat, how="vertical")
    # Keep a stable column order for downstream callers/UI.
    ordered_cols = [c for c in ["index", "bt_index", "seat", "Auction", "Expr",
                                "Agg_Expr_Seat_1", "Agg_Expr_Seat_2", "Agg_Expr_Seat_3", "Agg_Expr_Seat_4"] if c in out.columns]
    return out.select(ordered_cols)


def process_opening_bids_from_bt_seat1(
    deal_df: pl.DataFrame,
    bt_seat1_df: pl.DataFrame,
    deal_criteria_by_seat_dfs: Dict[int, Dict[str, pl.DataFrame]],
    seats: List[int] | None = None,
    directions: List[str] | None = None,
    opening_directions: List[str] | None = None,
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], pl.DataFrame]:
    """Compute opening-bid candidates using only `bt_seat1_df`.
    
    Returns:
    - results: same structure as `process_opening_bids(...)` (candidates + original_indices)
    - bt_openings_df: small lookup table used by the API for display/filtering
    """
    bt_openings_df = build_opening_bids_table_from_bt_seat1(bt_seat1_df)

    elapsed_time = time.time()
    results: Dict[Tuple[str, int], Dict[str, Any]] = {}

    seats_to_process = seats if seats is not None else [1, 2, 3, 4]
    directions_to_process = directions if directions is not None else DIRECTIONS

    valid_combos: set[Tuple[str, int]] = set()
    if opening_directions is not None:
        for dealer in directions_to_process:
            dealer_idx = DIRECTIONS.index(dealer)
            for seat in seats_to_process:
                opener = DIRECTIONS[(dealer_idx + seat - 1) % 4]
                if opener in opening_directions:
                    valid_combos.add((dealer, seat))
    else:
        for dealer in directions_to_process:
            for seat in seats_to_process:
                valid_combos.add((dealer, seat))

    print(f"Processing {len(valid_combos)} (dealer, seat) combinations (seat1-only openings)...")
    openings_by_seat: Dict[int, pl.DataFrame] = {
        s: bt_openings_df.filter(pl.col("seat") == s) for s in seats_to_process
    }

    for dealer in directions_to_process:
        dealer_mask = deal_df["Dealer"] == dealer
        if not dealer_mask.any():
            print(f"Skipping Dealer={dealer} (no deals in dataset)")
            continue

        dealer_indices = dealer_mask.arg_true()

        for seat in seats_to_process:
            if (dealer, seat) not in valid_combos:
                continue

            bt_openings_df_for_seat = openings_by_seat.get(seat)
            if bt_openings_df_for_seat is None:
                continue

            criteria = deal_criteria_by_seat_dfs[seat][dealer][dealer_indices]

            t = time.time()
            new_candidates = find_all_opening_bids_by_seat(criteria, bt_openings_df_for_seat, seat)
            elapsed = max(time.time() - t, 1e-6)

            direction = get_direction_for_seat(dealer, seat)
            print(
                f"Dealer={dealer}, Seat {seat} ({direction}): {elapsed:.2f}s, "
                f"{criteria.height / elapsed:.0f}/sec, shape={new_candidates.shape}"
            )

            results[(dealer, seat)] = {
                "candidates": new_candidates,
                "original_indices": dealer_indices,
            }

            del criteria
            gc.collect()

    total_elapsed = time.time() - elapsed_time
    print(f"\nTotal result keys: {len(results)} in {total_elapsed:.1f}s")
    return results, bt_openings_df
