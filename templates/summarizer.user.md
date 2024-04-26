The user's latest interaction with the Interface bot have been:

<% for message in latest_messages %>
[{{message.role}}]: {{message.content}}
<% endfor %>

Determine whether this excerpt will help you answer the user's question. If it will, reply with the following yaml:

```yaml
relevant: true
summary: <summary of the excerpt>
complete: <true or false, whether it suffices to answer the question>
```

Otherwise, reply with the following yaml:

```yaml
relevant: false
```
