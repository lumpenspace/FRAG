The user's latest interaction with the Interface bot have been:

<% for message in latest_messages %>
[{{message.role}}]: {{message.content}}
<% endfor %>

Determine whether this excerpt will help you answer the user's question. If it will, reply with the following yaml:

```xml
<relevant>true</relevant>
<complete>true</complete><!-->If it answer the question by itself, otherwise false<-->
<summary>The summary of the relevant parts</summary>
```

Otherwise, reply with the following yaml:

```xml
<relevant>false</relevant>
```