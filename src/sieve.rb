#!/usr/bin/env ruby
# -*- encoding: utf-8 -*-

def print_help
	puts "Usage: #{__FILE__} in_file [out_file] [sieve_size=50]" 
end

if ARGV.size < 1
  print_help
  exit 1
end

in_file_name = ARGV[0]

if ARGV.size > 1
  out_file_name = ARGV[1]
else
  out_file_name = in_file_name + "_new"
end

if ARGV.size > 2
  sieve_size = Integer(ARGV[2])
else
  sieve_size = 50
end

i=0
File.open(out_file_name, 'w') do |out_file|
  File.open(in_file_name, 'r') do |in_file|
    while line = in_file.gets
      out_file.puts line if i % sieve_size == 0
      i+=1
    end
  end
end

puts "Generated file: #{out_file_name}"
