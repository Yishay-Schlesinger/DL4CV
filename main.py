from escnn.group import groups

G = groups.so3_group(maximum_frequency=6)
group_element = G.sample()

print(group_element)