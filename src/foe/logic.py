from dataclasses import dataclass
from typing import Tuple, Union
from lark import Lark, ParseTree
from itertools import combinations


@dataclass(eq=True, frozen=True)
class Function:
    name: str
    arguments: Tuple['Term']

    def __repr__(self) -> str:
        if len(self.arguments) == 0:
            return self.name
        else:
            return "{}({})".format(
                self.name,
                ", ".join(repr(a) for a in self.arguments),
            )

    def __lt__(self, other):
        return (self.name, self.arguments) < (other.name, other.arguments)


@dataclass(eq=True, frozen=True)
class Variable:
    id: int

    def __repr__(self) -> str:
        return "?{}".format(self.id)

    def __lt__(self, other):
        return self.id < other.id


Term = Function | Variable


def is_subterm(subterm: Term, term: Term) -> bool:
    """Checks if a term is a subterm of another term.

    Parameters
    ----------
    Term : Term
        The term.
    subterm : Term
        The subterm.

    Returns
    -------
    bool
        True if the subterm is a subterm of the term, False otherwise.

    Example
    -------
    >>> t = Function("f", (Variable(0), Variable(1)))
    >>> is_subterm(t, t)
    True
    >>> is_subterm(Variable(0), t)
    True
    >>> is_subterm(Variable(1), t)
    True
    >>> is_subterm(Function("f", (Variable(0), Variable(1))), t)
    True
    >>> is_subterm(Function("f", (Variable(0), Variable(0))), t)
    False
    """
    if subterm == term:
        return True
    if isinstance(term, Function):
        for a in term.arguments:
            if is_subterm(subterm, a):
                return True
    return False


def get_subterm(Term, index: Tuple[int]):
    """Gets a subterm of a term.

    Parameters
    ----------
    Term : Term
        The term.
    index : Tuple[int]
        The index of the subterm.

    Returns
    -------
    Term
        The subterm.

    Example
    -------
    >>> t = Function("f", (Variable(0), Variable(1)))
    >>> subterm(t, (0,))
    Variable(0)
    >>> subterm(t, (1,))
    Variable(1)
    >>> subterm(t, (0, 0))
    Traceback (most recent call last):
    ...
    TypeError: 'Variable' object is not subscriptable
    """
    for i in index:
        Term = Term.arguments[i]
    return Term


def replace_subterm(s: Term, index: tuple[int, ...], t: Term):
    """Replaces a subterm of a term.

    Parameters
    ----------
    s : Term
        The term.
    index : tuple[int, ...]
        The index of the subterm.
    t : Term
        The term to replace the subterm with.

    Returns
    -------
    Term
        The term with the subterm replaced.

    Example
    -------
    >>> t = Function("f", (Variable(0), Variable(1)))
    >>> replace_subterm(t, (0,), Variable(2))
    Function('f', (Variable(2), Variable(1)))
    >>> replace_subterm(t, (1,), Variable(2))
    Function('f', (Variable(0), Variable(2)))
    """
    if len(index) == 0:
        return t
    elif len(index) == 1:
        if isinstance(s, Function):
            return Function(s.name, tuple(
                t if i == index[0] else a
                for i, a in enumerate(s.arguments)
            ))
        else:
            raise TypeError("'{}' object is not subscriptable".format(
                type(s).__name__
            ))
    else:
        if isinstance(s, Function):
            return Function(s.name, tuple(
                replace_subterm(a, index[1:], t)
                if i == index[0] else a
                for i, a in enumerate(s.arguments)
            ))
        else:
            raise TypeError("'{}' object is not subscriptable".format(
                type(s).__name__
            ))


@dataclass(eq=True, frozen=True)
class Equation:
    lhs: Term
    rhs: Term

    def __repr__(self) -> str:
        return "{} = {}".format(self.lhs, self.rhs)

    def __lt__(self, other):
        return (self.lhs, self.rhs) < (other.lhs, other.rhs)


@dataclass(eq=True, frozen=True)
class Sequent:
    antecedent: Tuple[Equation]
    succedent: Tuple[Equation]

    def __repr__(self) -> str:
        return "{} ⊢ {}".format(
            ", ".join(repr(e) for e in self.antecedent),
            ", ".join(repr(e) for e in self.succedent),
        )

    def __lt__(self, other):
        return (
            self.antecedent, self.succedent
        ) < (
            other.antecedent, other.succedent
        )

    def equality_resolution(self) -> 'Sequent':
        """Applies equality resolution to the sequent.

        Returns
        -------
        Sequent
            The sequent resulting from applying equality resolution.
        """
        for equation in self.antecedent:
            unifier = mgu(equation.lhs, equation.rhs, disjoint=False)
            if unifier is not None:
                return Sequent(
                    tuple(substitute(
                        unifier, e
                    ) for e in self.antecedent if e != equation),
                    tuple(substitute(
                        unifier, e
                    ) for e in self.succedent),
                )
        return self

    def equality_factoring(self) -> 'Sequent':
        """Applies equality factoring to the sequent.

        Returns
        -------
        Sequent
            The sequent resulting from applying equality factoring.
        """
        for (e1, e2) in combinations(self.succedent, 2):
            for (t1, t2, t3, t4) in (
                (e1.lhs, e1.rhs, e2.lhs, e2.rhs),
                (e1.rhs, e1.lhs, e2.lhs, e2.rhs),
                (e1.lhs, e1.rhs, e2.rhs, e2.lhs),
                (e1.rhs, e1.lhs, e2.rhs, e2.lhs)
            ):
                unifier = mgu(t1, t3, disjoint=False)
                if unifier is not None:
                    return Sequent(
                        tuple(substitute(
                            unifier, e
                        ) for e in self.antecedent) +
                        (Equation(
                            substitute(unifier, t2),
                            substitute(unifier, t4),
                        ),),
                        tuple(substitute(
                            unifier, e
                        ) for e in self.succedent if e != e1 and e != e2) +
                        (Equation(
                            substitute(unifier, t1),
                            substitute(unifier, t4),
                        ),),
                    )
        return self

    def normalize(self) -> 'Sequent':
        """Normalizes the sequent.

        Returns
        -------
        Sequent
            The normalized sequent.
        """
        return self.equality_factoring().equality_resolution()


Substitution = dict[Variable, Term]


def compose_substitutions(s1: Substitution, s2: Substitution) -> Substitution:
    """Composes two substitutions.

    Parameters
    ----------
    s1 : Substitution
        The first substitution.
    s2 : Substitution
        The second substitution.

    Returns
    -------
    Substitution
        The composition of the two substitutions.

    Example
    -------
    >>> s1 = {Variable(0): Variable(1)}
    >>> s2 = {Variable(1): Variable(2)}
    >>> compose_substitutions(s1, s2)
    {Variable(0): Variable(2)}
    """
    return {k: substitute(s2, v) for k, v in s1.items()}


def mgu(t1: Term, t2: Term, disjoint=True) -> Substitution:
    """Computes the most general unifier of two terms.

    Parameters
    ----------
    t1 : Term
        The first term.
    t2 : Term
        The second term.
    disjoint : bool, optional
        If True treats variables as disjoint, even if they have the same id.
        By default True.

    Returns
    -------
    Substitution
        The most general unifier of the two terms.
        None if the terms cannot be unified.

    Example
    -------
    >>> t1 = Function("f", (Variable(0),))
    >>> t2 = Function("f", (Variable(1),))
    >>> mgu(t1, t2)
    ({Variable(0): Variable(1)}, {})
    """
    if isinstance(t1, Variable):
        if isinstance(t2, Variable):
            if t1 == t2:
                if disjoint:
                    return (dict(), dict())
                else:
                    return dict()
            else:
                if disjoint:
                    return ({t1: t2}, dict())
                else:
                    return dict()
        else:
            if disjoint:
                return ({t1: t2}, dict())
            else:
                if is_subterm(t1, t2):
                    return None
                else:
                    return {t1: t2}
    elif isinstance(t2, Variable):
        if disjoint:
            return ({t2: t1}, dict())
        else:
            if is_subterm(t2, t1):
                return None
            else:
                return {t2: t1}
    else:
        if t1.name != t2.name:
            return None
        elif len(t1.arguments) != len(t2.arguments):
            return None
        else:
            if disjoint:
                s1 = dict()
                s2 = dict()
                for a1, a2 in zip(t1.arguments, t2.arguments):
                    m = mgu(substitute(s1, a1), substitute(s2, a2))
                    if m is None:
                        return None
                    s1, s2 = m
                return (s1, s2)
            else:
                s = dict()
                for a1, a2 in zip(t1.arguments, t2.arguments):
                    s = mgu(substitute(s, a1), substitute(s, a2))
                    if s is None:
                        return None
                return s


def substitute(
    s: Substitution,
    x: Union[Term, Equation, Sequent]
) -> Union[Term, Equation, Sequent]:
    """Substitutes a term for a variable.

    Parameters
    ----------
    s : Substitution
        The substitution.
    x : Union[Term, Equation, Sequent]
        The term, equation or sequent.

    Returns
    -------
    Union[Term, Equation, Sequent]
        The term, equation or sequent with the substitution applied.

    Example
    -------
    >>> s = {Variable(0): Variable(1)}
    >>> x = Equation(Variable(0), Variable(2))
    >>> substitute(s, x)
    Equation(Variable(1), Variable(2))
    """
    if isinstance(x, Equation):
        return Equation(substitute(s, x.lhs), substitute(s, x.rhs))
    elif isinstance(x, Sequent):
        return Sequent(
            tuple(substitute(s, e) for e in x.antecedent),
            tuple(substitute(s, e) for e in x.succedent)
        )
    else:
        if isinstance(x, Variable):
            if x in s:
                return s[x]
            else:
                return x
        else:
            return Function(
                x.name,
                tuple(substitute(s, a) for a in x.arguments)
            )


class Problem():
    """An environment for first-order logic.

    Attributes
    ----------
    sorts : set[str]
        The sorts that have been declared.
    functions : set[(str, Tuple[str], str)]
        The functions that have been declared.
    function_sorts : dict[str, (Tuple[str], str)]
        A mapping from function names to their argument and result sorts.
    sequents : list[Sequent]
        The sequents that have been read.
    variablecounter : int
        The number of variables that have been declared.
    variablessorts : dict[Variable, str]
        A mapping from variables to their sorts.

    Methods
    -------
    declare_sort(s: str)
        Declares a new sort.
    declare_function(name: str, argument_sorts: Tuple[str], result_sort: str)
        Declares a new function.
    read_sequent(s: str)
        Parses a sequent from a string.
    """

    def __init__(self):
        self.sorts: set[str] = set()
        self.functions: set[(str, Tuple[str], str)] = set()
        self.function_sorts: dict[str, (Tuple[str], str)] = dict()
        self.sequents: list[Sequent] = list()
        self.variablecounter: int = 0
        self.variablessorts: dict[Variable, str] = dict()

    def declare_sort(self, s: str):
        """Declares a new sort.

        Parameters
        ----------
        s : str
            The name of the sort.

        Example
        -------
        >>> env = Environment()
        >>> env.declare_sort("S")

        """
        if s in self.sorts:
            raise Exception(f"Sort {s} already exists!")
        else:
            self.sorts.add(s)

    def declare_function(
        self,
        name: str,
        argument_sorts: Tuple[str],
        result_sort: str
    ):

        """Declares a new function.

        Parameters
        ----------
        name : str
            The name of the function.
        argument_sorts : Tuple[str]
            The sorts of the arguments.
        result_sort : str
            The sort of the result.

        Example
        -------
        >>> env = Environment()
        >>> env.declare_sort("S")
        >>> env.declare_function("f", ("S",), "S")
        """
        if name in self.functions:
            raise Exception(f"Function {name} already exists!")
        for sort in argument_sorts:
            if sort not in self.sorts:
                raise Exception(f"Sort {sort} does not exist!")
        if result_sort not in self.sorts:
            raise Exception(f"Sort {result_sort} does not exist!")
        self.functions.add(name)
        self.function_sorts[name] = (argument_sorts, result_sort)
        newline = "\n"
        self.grammar = Lark(f'''
            %import common.WS
            %ignore WS
            %import common.CNAME
            sequent: equations "->" equations
            equations: (equation ("," equation)*)?
            equation: term "=" term
            term: {" | ".join(f'{f}' for f in  self.functions)} | variable
            {newline.join(f'{f}: "{f}" arguments' for f in  self.functions)}
            arguments: ("(" term ("," term)* ")")?
            variable: CNAME
        ''', start="sequent")

    def read_sequent(self, s: str):
        """Parses a sequent from a string.

        Parameters
        ----------
        s : str
            The string to parse.

        Example:
        >>> env = Environment()
        >>> env.declare_sort("S")
        >>> env.declare_function("f", ("S",), "S")
        >>> env.read_sequent("-> x = f(x)")
        """
        try:
            tree = self.grammar.parse(s)
        except Exception:
            raise Exception(f"Cannot parse sequent {s}")
        sequent = self._parse_sequent(tree)
        types = self._typecheck(sequent)
        substitution = dict()
        for id, sort in types.items():
            substitution[Variable(id)] = Variable(self.variablecounter)
            self.variablessorts[Variable(self.variablecounter)] = sort
            self.variablecounter += 1
        self.sequents.append(substitute(substitution, sequent))

    def _parse_sequent(self, tree: ParseTree) -> Sequent:
        antecedent = self._parse_equations(tree.children[0])
        succedent = self._parse_equations(tree.children[1])
        return Sequent(antecedent, succedent)

    def _parse_equations(self, tree) -> Tuple[Equation]:
        return tuple(self._parse_equation(e) for e in tree.children)

    def _parse_equation(self, tree) -> Equation:
        return Equation(
            self._parse_term(tree.children[0]),
            self._parse_term(tree.children[1])
        )

    def _parse_term(self, tree) -> Term:
        if tree.children[0].data.value == "variable":
            return Variable(tree.children[0].children[0].value)
        else:
            return Function(
                tree.children[0].data.value,
                self._parse_arguments(tree.children[0].children[0])
            )

    def _parse_arguments(self, tree) -> Tuple[Term]:
        if tree is None:
            return tuple()
        else:
            return tuple(self._parse_term(t) for t in tree.children)

    def _typecheck(self, sequent) -> dict:
        """Typechecks a sequent.

        Parameters
        ----------
        sequent : Sequent
            The sequent to typecheck.

        Returns
        -------
        dict
            A mapping from variables to sorts.

        Raises
        ------
        Exception
            If not all variables can be assigned a sort.

        Example:
        >>> env = Environment()
        >>> env.declare_sort("S")
        >>> env.declare_sort("T")
        >>> env.declare_function("f", ("S",), "T")
        >>> env.read_sequent("-> z = y, y = f(x)")
        >>> env.typecheck(env.sequents[0])
        {'z': 'T', 'y': 'T', 'x': 'S'}
        """
        variablessorts = dict()
        unresolved = sequent.antecedent + sequent.succedent
        while unresolved:
            resolved = 0
            equations = unresolved
            unresolved = list()
            for equation in equations:
                sort = self._typecheck_equation(equation, variablessorts)
                if sort is None:
                    unresolved.append(equation)
                else:
                    resolved += 1
            if resolved == 0:
                raise Exception("Cannot infer types of all variables")
        return variablessorts

    def _typecheck_equation(self, equation, variablesorts):
        lhs_sort = self._typecheck_term(equation.lhs, variablesorts, None)
        rhs_sort = self._typecheck_term(equation.rhs, variablesorts, lhs_sort)
        if lhs_sort != rhs_sort:
            lhs_sort = self._typecheck_term(
                equation.lhs, variablesorts, rhs_sort)
        return lhs_sort

    def _typecheck_term(self, term, variablesorts, sort):
        if isinstance(term, Variable):
            if term.id not in variablesorts:
                if sort is None:
                    return None
                else:
                    variablesorts[term.id] = sort
                    return sort
            else:
                if sort is not None and variablesorts[term.id] != sort:
                    raise Exception(f"Sort of {term.id} is ambiguous!")
                else:
                    return variablesorts[term.id]
        else:
            if term.name not in self.function_sorts:
                raise Exception(f"Function {term.name} does not exist!")
            else:
                arg_sorts, result_sort = self.function_sorts[term.name]
                if len(term.arguments) != len(arg_sorts):
                    raise Exception(
                        f"Function {term.name} has {len(arg_sorts)}\
                        arguments but {len(term.arguments)} were given!"
                    )
                for arg, arg_sort in zip(term.arguments, arg_sorts):
                    self._typecheck_term(arg, variablesorts, arg_sort)
                if sort is not None and result_sort != sort:
                    raise Exception(
                        f"Function {term.name} has sort {result_sort}\
                        but must have sort {sort}!"
                    )
                return result_sort
