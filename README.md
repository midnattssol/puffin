# puffin

Puffin is a code editor built in Python with Shortcake/Gdk.

# TODO

## Features

[x] Working noun/verb system
[.] Tags

### Verbs

[x] d(elete)
    [.] dd: delete current line
    [.] d[, d{, d(, d', d": delete currently selected item enclosed by these symbols
[x] p(aste)
[x] hjkl: normal movement
[x] w: word forwards
[.] g(o to start of line)
    [.] g[n]: go to line x, supporting negative indexing
[.] r: find and replace
    [.] avoid autoenter
    [.] r[n=1]?l "..." "..."? -> match and (replace?) n
    [.] rs "..." -> select all

## Other


[.] UI
    [x] Bug: Lines added with Insert mode and Enter can't be jumped to
    [x] Focused line in middle
    [ ] Scrolling free of cursor
    [x] Scolling with up/down keys
        [x] Bug: Always lands on last char of line
    [x] Color system
        [ ] Show selections with background square thing
        [ ] Syntax highlighting
        [ ] Parenthesis highlighting
        [ ] Choose if lines start at 0 or 1

[.] Input
    [x] Bug: Input on other line than 0th inputs on 0th line either way
    [x] Implement up and down arrows correctly
    [x] Working cursor
    [x] Changing modes
        [.] Vimesque command line mode
            [ ] Multicursor capabilities
        [x] Parsing more complex instructions
            [.] Selections
[ ] IO capabilities
    [ ] Stdout/stderr display


[ ] Ability to use Bash?
    - One problem might be that supporting ANSI escape codes is really complicated
    - That as well as showing new lines as they are printed to stdout might be weird

[ ] Snippet system
    - When in insert mode, snippets should show up as autocompletions under the current text.
    - When in command/shell mode, it would be nice to show continuations and be able to autocomplete them to run a previous command again.
    - That and being able to use the arrow keys Up Down to scroll snippets in the buffer and Tab to autocomplete (or Right in the command shell)


[ ] Set up system to easily change keybindings to trigger custom scripts

    [ ] b(ind) [str keybind]
        - Should open the file puffin/keybinds/$escaped_keybind.py
        - Access to the current buffer
        - Python script with access to the current buffer
        - Live reloading would be nice
            - Just check file saves, and if the file saved is one of those, import that instead. The problem with this approach is that it only works if the text is edited in Puffin.
            - Alternatively, fork the process into a watching process which tells the main process when the file has changed and it's time to import the thing again. This is probably more solid.

        - a keybind is a function f: buffer -> {None} which usually manipulates the current buffer or fetches data from it, returning it to stdout.


## Maintainability

[ ] Clean the editor class
