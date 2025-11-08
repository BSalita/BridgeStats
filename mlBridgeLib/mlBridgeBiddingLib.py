
# mlBridgeBiddingLib.py

import polars as pl
import numpy as np
import tqdm
import re
import operator
from asteval import Interpreter
from collections import defaultdict
import time


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
    

    def handle_chained_comparison(self, tokens):
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
            variables.update(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
        return list(variables)


    def create_parsing_logic(self, criteria_l):
        """Create parsing logic for a list of infix expressions."""
        variables = self.extract_variables(criteria_l)
        postfix_expressions = []
        
        for expr in criteria_l:
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
        expressions = sorted(set(expressions_l))
        print(f"create_eval_criteria: done: time: {time.time()-t:.2f}") # 1s
        return expressions


    def create_directional_exprs_cols(self, criteria_variables_infix_index_to_sym_d):
        self.directional_exprs_cols_d = {} # map of original expr to directional expr
        self.directional_variable_cols_d = {} # map of original expr to extracted directional variable
        self.regex_col_names_d = {}
        for k,v in criteria_variables_infix_index_to_sym_d.items():
            splits_space = v.split(' ')
            directional_var_name = splits_space[0]
            splits_underscore = splits_space[0].split('_')
            if len(splits_space) == 1:
                assert len(splits_underscore) > 1, v
                if splits_underscore[-1] in 'SHDCN':
                    var_name = '_'.join(splits_underscore[:-1])+v[-2:].replace('_C','_{d}_C').replace('_D','_{d}_D').replace('_H','_{d}_H').replace('_S','_{d}_S').replace('_N','_{d}_N')
                    self.directional_variable_cols_d[directional_var_name] = var_name
                    self.directional_exprs_cols_d[v] = var_name
                    self.regex_col_names_d[directional_var_name] = var_name[:-1].replace('{d}_','([NESW]_)?[CDHS]') 
                elif splits_underscore[0] == 'C':
                    var_name = v.replace('C_','C_{d}')
                    self.directional_variable_cols_d[directional_var_name] = var_name
                    self.directional_exprs_cols_d[v] = var_name
                    self.regex_col_names_d[directional_var_name] = var_name[:-2].replace('{d}','([NESW])?[CDHS][AKQJT98765432]') # might need to add JT9-2
                else:
                    print('Ignoring non-directional:',v)
                    var_name = v
                    self.directional_variable_cols_d[directional_var_name] = var_name
                    self.directional_exprs_cols_d[v] = var_name
                    self.regex_col_names_d[directional_var_name] = var_name
            else:
                if splits_underscore[-1] in 'SHDCN':
                    var_name = '_'.join(splits_underscore[:-1]+['{d}',splits_underscore[-1]])
                    self.directional_variable_cols_d[directional_var_name] = var_name
                    self.directional_exprs_cols_d[v] = var_name+' '+' '.join(splits_space[1:])
                    self.regex_col_names_d[directional_var_name] = var_name.replace('{d}','([NESW]_)?[CDHS]')
                else:
                    var_name = splits_space[0]+'_{d}'
                    self.directional_variable_cols_d[directional_var_name] = var_name
                    self.directional_exprs_cols_d[v] = var_name+' '+' '.join(splits_space[1:])
                    self.regex_col_names_d[directional_var_name] = var_name.replace('{d}','([NESW]_)?[CDHS]')
        return # self.regex_col_names_d, self.directional_variable_cols_d, self.directional_exprs_cols_d


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
        self.create_directional_exprs_cols(self.criteria_variables_infix_index_to_sym_d)
        
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
        self.first_bidder_of_strain_d = {(None, None): None}

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


