# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Protocols::ChatCompletions::Streaming do
  let(:protocol) { RubyLLM::Protocols::ChatCompletions.allocate }

  it 'splits thinking chunk deltas from content deltas' do
    thinking_chunk = protocol.send(
      :build_chunk,
      {
        'model' => 'mistral-small-latest',
        'choices' => [
          {
            'delta' => {
              'content' => [
                { 'type' => 'thinking', 'thinking' => [{ 'type' => 'text', 'text' => 'pondering' }], 'closed' => true },
                { 'type' => 'text', 'text' => 'Four' }
              ]
            }
          }
        ]
      }
    )

    expect(thinking_chunk.thinking.text).to eq('pondering')
    expect(thinking_chunk.content).to eq('Four')

    text_chunk = protocol.send(
      :build_chunk,
      {
        'model' => 'mistral-small-latest',
        'choices' => [{ 'delta' => { 'content' => ' and more' } }]
      }
    )

    expect(text_chunk.thinking).to be_nil
    expect(text_chunk.content).to eq(' and more')
  end

  it 'leaves tokens unknown on chunks that report no usage' do
    chunk = protocol.send(
      :build_chunk,
      { 'model' => 'mistral-small-latest', 'choices' => [{ 'delta' => { 'content' => 'Four' } }] }
    )

    expect(chunk.tokens.to_h).to be_empty
  end

  it 'streams refusal deltas as content' do
    chunk = protocol.send(
      :build_chunk,
      {
        'model' => model_for(:openai, :temperature),
        'choices' => [{ 'delta' => { 'content' => nil, 'refusal' => 'I cannot help' } }]
      }
    )

    expect(chunk.content).to eq('I cannot help')
  end

  it 'preserves raw finish reasons on chunks' do
    chunk = protocol.send(
      :build_chunk,
      {
        'model' => 'gpt-4.1-nano',
        'choices' => [
          {
            'delta' => { 'content' => '' },
            'finish_reason' => 'tool_calls'
          }
        ]
      }
    )

    expect(chunk.finish_reason).to eq(:tool_calls)
  end

  describe '#parse_streaming_error' do
    it 'parses typed error objects' do
      status, message = protocol.send(
        :parse_streaming_error,
        { error: { type: 'rate_limit_exceeded', message: 'Slow down' } }.to_json
      )

      expect(status).to eq(429)
      expect(message).to eq('Slow down')
    end

    it 'reports a 500 for server errors' do
      status, message = protocol.send(
        :parse_streaming_error,
        { error: { type: 'server_error', message: 'Internal error' } }.to_json
      )

      expect(status).to eq(500)
      expect(message).to eq('Internal error')
    end

    it 'reads the HTTP status from a numeric code' do
      status, message = protocol.send(
        :parse_streaming_error,
        { error: { code: 502, message: 'Provider returned error' } }.to_json
      )

      expect(status).to eq(502)
      expect(message).to eq('Provider returned error')
    end

    it 'reports a 401 for authentication errors' do
      status, = protocol.send(
        :parse_streaming_error,
        { error: { type: 'authentication_error', message: 'Authentication Fails' } }.to_json
      )

      expect(status).to eq(401)
    end

    it 'reports the server-side status for overloaded and unavailable services' do
      statuses = %w[overloaded_error service_unavailable].map do |type|
        protocol.send(:parse_streaming_error, { error: { type:, message: 'Try again later' } }.to_json).first
      end

      expect(statuses).to eq([529, 503])
    end

    it 'falls back to a 400 for other typed error objects' do
      status, message = protocol.send(
        :parse_streaming_error,
        { error: { type: 'invalid_request_error', message: 'Bad request' } }.to_json
      )

      expect(status).to eq(400)
      expect(message).to eq('Bad request')
    end

    it 'handles a body that parses to a bare JSON string' do
      status, message = protocol.send(
        :parse_streaming_error,
        '"The model foo is not available in your region (error)."'
      )

      expect(status).to be_nil
      expect(message).to eq('The model foo is not available in your region (error).')
    end

    it 'handles a string error value' do
      status, message = protocol.send(
        :parse_streaming_error,
        { error: 'The model foo is not available in your region.' }.to_json
      )

      expect(status).to be_nil
      expect(message).to eq('The model foo is not available in your region.')
    end
  end

  it 'surfaces the provider message for a failed streaming response with a string error value' do
    provider = instance_double(RubyLLM::Providers::OpenAI, slug: 'openai', parse_error_code: nil)
    allow(provider).to receive(:parse_error) { |response| response.body['error'] }
    protocol.instance_variable_set(:@provider, provider)
    failed_env = Faraday::Env.from(status: 404)

    expect do
      protocol.send(
        :handle_failed_response,
        '{"error": "The model foo is not available in your region."}',
        +'',
        failed_env
      )
    end.to raise_error(RubyLLM::Error, /not available in your region/)
  end

  describe 'inline think tags' do
    let(:chunks) { [] }

    def stream_deltas(*deltas, target: protocol)
      allow(target).to receive(:stream_events) do |*, &block|
        deltas.each { |delta| block.call('model' => 'qwen3', 'choices' => [{ 'delta' => { 'content' => delta } }]) }
        instance_double(Faraday::Response)
      end
      target.send(:stream_response, {}) { |chunk| chunks << chunk }
    end

    it 'splits think tags into thinking, even when a tag spans chunks' do
      message = stream_deltas('<thi', 'nk>reason', 'ing</th', "ink>\nans", 'wer')

      expect(message.content).to eq('answer')
      expect(message.thinking.text).to eq('reasoning')
    end

    it 'yields the split to the stream block' do
      stream_deltas('<think>secret', '</think>', "\nanswer")

      expect(chunks.map(&:content).join).to eq('answer')
      expect(chunks.filter_map { |chunk| chunk.thinking&.text }.join).to eq('secret')
    end

    it 'keeps a lone angle bracket as content' do
      message = stream_deltas('a <', 'b')

      expect(message.content).to eq('a <b')
      expect(message.thinking).to be_nil
    end

    it 'treats a think tag after the start as plain text' do
      message = stream_deltas('Use the <think> tag ', 'to wrap reasoning. More answer.')

      expect(message.content).to eq('Use the <think> tag to wrap reasoning. More answer.')
      expect(message.thinking).to be_nil
    end

    it 'treats an unclosed think block as thinking' do
      message = stream_deltas('<think>cut off')

      expect(message.content).to be_nil
      expect(message.thinking.text).to eq('cut off')
    end

    it 'does not split on the Responses protocol' do
      expect(RubyLLM::Protocols::Responses.allocate.send(:split_think_tags?)).to be(false)
    end
  end
end
