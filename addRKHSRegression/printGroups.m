function printGroups(groups)
  str = sprintf('Num Groups: %d\n', numel(groups));
  for i = 1:numel(groups)
%     groups{i},
    str = sprintf('%s%s\n', str, mat2str(groups{i}) );
  end
  fprintf(str);
end
