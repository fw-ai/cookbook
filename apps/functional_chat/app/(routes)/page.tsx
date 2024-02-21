import { ChatInferenceModule } from '~/components/chat/ChatInferenceModule';
/* eslint camelcase: 0 */
import { cn } from '~/lib/utils';

// eslint-disable-next-line complexity
export default async function Page() {
    return (
        <>
            <div className="flex">
                <div className={cn('flex flex-col', 'sm:container')}>
                    <div className="flex-row flex">
                        <div className="grow">
                            <ChatInferenceModule />
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
