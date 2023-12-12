    @torch.inference_mode()
    def batch_inference(
        self,
        texts,  # List of texts
        language,
        gpt_cond_latent,  # Assuming this is a batch of latents
        speaker_embeddings,  # Assuming this is a batch of speaker embeddings

        temperature=0.65,
        length_penalty=1,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,

        **hf_generate_kwargs
    ):
        
        print("Inference: Shape of gpt_cond_latent is : ", gpt_cond_latent.shape)
        print("Inference : Shape of speaker embeddings  :" , speaker_embeddings.shape)
        
        batch_size = len(texts)
        max_length = max(len(text) for text in texts)

        # Handling text inputs for batch

        #text_tokens = torch.IntTensor(self.tokenizer.encode(text, lang=language)).unsqueeze(0).to(self.device)
        for text in texts:
            text_tokens = [torch.IntTensor(self.tokenizer.encode(text.strip().lower(), lang=language))]

        #print("Shape of text tokens that are generate",text_tokens.shape)

        for i, tokens in enumerate(text_tokens):
            print(f"Shape of text tokens {i}: {tokens.shape}")

        text_tokens_padded = torch.nn.utils.rnn.pad_sequence(
            text_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)

        # Verifying token length constraint for each text
        for tokens in text_tokens:
            assert (
                tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

        # Batch inference steps (similar to original but adjusted for batch)
        with torch.no_grad():
            gpt_codes_batch = [
                # Process each item in the batch separately
                self.gpt.generate(
                    cond_latents=gpt_cond_latent[i].unsqueeze(0),
                    text_inputs=text_tokens_padded[i].unsqueeze(0),
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs
                )
                for i in range(batch_size)
            ]

            expected_output_len_batch = [
                torch.tensor([gpt_codes.shape[-1] * self.gpt.code_stride_len], device=self.device)
                for gpt_codes in gpt_codes_batch
            ]

            print("Expected output length", expected_output_len_batch)

            text_len_batch = torch.tensor([tokens.shape[-1] for tokens in text_tokens], device=self.device)
            print("text length batch is", text_len_batch)

            gpt_latents_batch = [
                self.gpt(
                    text_tokens_padded[i].unsqueeze(0),
                    text_len_batch[i].unsqueeze(0),
                    gpt_codes_batch[i],
                    expected_output_len_batch[i],
                    cond_latents=gpt_cond_latents[i].unsqueeze(0),
                    return_attentions=False,
                    return_latent=True,
                )
                for i in range(batch_size)
            ]
            print("GPT latents are", gpt_latents_batch)

            wav_batch = [
                self.hifigan_decoder(gpt_latents, g=speaker_embeddings[i].unsqueeze(0))
                for i, gpt_latents in enumerate(gpt_latents_batch)
            ]
            print("Wav batch", wav_batch , wav_batch.shape)


        return [
            {
                "wav": wav.cpu().numpy().squeeze(),
                "gpt_latents": gpt_latents,
                "speaker_embedding": speaker_embeddings[i]
            }
            for i, (wav, gpt_latents) in enumerate(zip(wav_batch, gpt_latents_batch))
        ]