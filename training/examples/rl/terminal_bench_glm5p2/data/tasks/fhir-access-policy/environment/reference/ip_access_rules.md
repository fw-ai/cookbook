# IP Access Rules

One valuable way to prevent unauthorized data access is to restrict access to clinical data by the user's IP address.

Medplum customers primarily use IP Access Restrictions to ensure that users are only accessing data from approved origins such as on-premise devices or corporate VPN.

## Setting Up IP Address Restrictions

To configure IP address restrictions in Medplum, select an existing AccessPolicy and scroll down to the section labeled "IP Access Rules". You can add "allow" and "block" rules based on IP addresses.

The rules are evaluated sequentially until a matching rule is found. To effectively restrict access, start by specifying a series of "allow" rules for the desired IP addresses or IP address ranges.

Once you have specified all the "allow" rules, add a wildcard "block" rule at the end to block all other IP addresses. To do this, use an asterisk (*) as the value for the "block" rule.

Please note that only IPv4 IP addresses are supported, and partial IP addresses can be used for matching. For example, specifying the value "8.8." would match "8.8.8.8" but would not match "8.7.8.8".
