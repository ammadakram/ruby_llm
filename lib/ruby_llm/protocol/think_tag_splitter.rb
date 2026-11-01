# frozen_string_literal: true

module RubyLLM
  class Protocol
    class ThinkTagSplitter # :nodoc:
      OPEN = '<think>'
      CLOSE = '</think>'

      def initialize
        @state = :start
        @buffer = +''
      end

      def push(text)
        @buffer << text
        content = +''
        thinking = +''
        advance(content, thinking)
        [content, thinking]
      end

      def finish
        content = +''
        thinking = +''
        (@state == :thinking ? thinking : content) << @buffer
        @buffer = +''
        [content, thinking]
      end

      private

      def advance(content, thinking)
        loop do
          break if @buffer.empty?
          break unless send(:"step_#{@state}?", content, thinking)
        end
      end

      def step_start?(content, _thinking)
        stripped = @buffer.lstrip
        return false if stripped.empty? || (OPEN.start_with?(stripped) && stripped != OPEN)

        if stripped.start_with?(OPEN)
          @buffer = stripped.delete_prefix(OPEN)
          @state = :thinking
        else
          content << @buffer
          @buffer = +''
          @state = :content
        end
        true
      end

      def step_thinking?(_content, thinking)
        index = @buffer.index(CLOSE)
        if index
          thinking << @buffer[0...index]
          @buffer = @buffer[(index + CLOSE.length)..]
          @state = :after
          return true
        end

        held = (CLOSE.length - 1).downto(1).find { |length| @buffer.end_with?(CLOSE[0, length]) } || 0
        thinking << @buffer[0, @buffer.length - held]
        @buffer = @buffer[@buffer.length - held, held]
        false
      end

      def step_after?(_content, _thinking)
        @buffer = @buffer.lstrip
        @state = :content unless @buffer.empty?
        !@buffer.empty?
      end

      def step_content?(content, _thinking)
        content << @buffer
        @buffer = +''
        false
      end
    end
  end
end
